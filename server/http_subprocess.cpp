#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#define CPPHTTPLIB_ZLIB_SUPPORT 0
#include "blackwell/httplib.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sys/wait.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Global metrics
static std::atomic<uint64_t> g_request_count{0};
static std::atomic<uint64_t> g_error_count{0};
static std::atomic<uint64_t> g_total_latency_ms{0};
static std::chrono::steady_clock::time_point g_start_time = std::chrono::steady_clock::now();

// Rate limiter: simple token bucket
class RateLimiter {
    std::mutex mtx;
    std::chrono::steady_clock::time_point last;
    int tokens;
    int max_tokens;
    std::chrono::milliseconds refill_interval;
public:
    RateLimiter(int max, int per_sec) : last(std::chrono::steady_clock::now()), tokens(max), max_tokens(max), refill_interval(1000 / per_sec) {}
    bool allow() {
        std::lock_guard<std::mutex> g(mtx);
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last);
        if (elapsed >= refill_interval) {
            int refill = elapsed.count() / refill_interval.count();
            tokens = std::min(max_tokens, tokens + refill);
            last = now;
        }
        if (tokens > 0) { tokens--; return true; }
        return false;
    }
};
static RateLimiter g_rate_limiter(10, 5);  // 5 req/s burst 10

static void record_request(bool success, int64_t duration_ms) {
    g_request_count.fetch_add(1);
    if(!success) g_error_count.fetch_add(1);
    g_total_latency_ms.fetch_add((uint64_t)duration_ms);
}

// Blackwell HTTP Server
// Uses temp-file IPC: writes request JSON to /tmp/inf_{pid}.req,
// inference_server reads from that file instead of stdin.

#include "blackwell/bpe_tokenizer.h"

class LocalTokenizer {
    blackwell::BpeTokenizer tok_;
    bool ok_{false};
public:
    LocalTokenizer(const char* model) {
        // Build path to tokenizer in weight directory
        // model names: llama32-1b, llama31-8b, qwen3-8b, 1.7b, 8b, 9b, batched
        std::string td;
        struct { const char* key; const char* path; } tok_dirs[] = {
            {"llama32-1b", "/mnt/data/ai/models/llama32-1b-int4-from-safetensors/tokenizer_data.bin"},
            {"llama31-8b", "/mnt/data/ai/models/llama31-8b-int4-from-safetensors/tokenizer_data.bin"},
            {"gemma", "./weights_gemma/tokenizer_data.bin"},
            {"qwen3", "/mnt/data/ai/models/qwen3-8b-int4/tokenizer_data.bin"},
        };
        td = "./tokenizer_data.bin";
        for (auto& d : tok_dirs) { if (strstr(model, d.key)) { td = d.path; break; } }
        ok_ = tok_.load(td.c_str()) == 0;
        fprintf(stderr, "[LocalTokenizer] ok=%d\n", ok_);
    }
    std::string decode(const std::vector<uint32_t>& ids) {
        std::string s;
        for(uint32_t id : ids) s += tok_.decode(id);
        return s;
    }
    explicit operator bool() const { return ok_; }
};

class SubprocessEngine {
    pid_t pid{-1};
    int wfd{-1};
    FILE* from_f{nullptr};
public:
    FILE* get_stream() { return from_f; }
    int get_write_fd() { return wfd; }
    pid_t get_pid() { return pid; }
    std::mutex lock;
    bool ready{false};
    bool is_gemma{false};    // Write all bytes to fd, retrying on short writes
    bool write_all(int fd, const char* buf, size_t len) {
        while (len > 0) {
            ssize_t n = write(fd, buf, len);
            if (n < 0 && errno != EINTR) return false;
            if (n < 0) continue; // EINTR retry
            buf += n; len -= n;
        }
        return true;
    }
    // Write string to temp file then pipe it to subprocess stdin
    // JSON-escape a string for embedding in a JSON value
    static std::string escape_json(const std::string& s) {
        std::string r;
        for (unsigned char c : s) {
            if (c == '\\') r += "\\\\";
            else if (c == '"') r += "\\\"";
            else if (c == '\n') r += "\\n";
            else if (c == '\r') r += "\\r";
            else if (c == '\t') r += "\\t";
            else if (c < 0x20 || c == 0x7f) r += ' ';
            else if (c >= 0x80) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); r += b; }
            else r += (char)c;
        }
        return r;
    }
    bool send_request(const char* req, int len) {
        char tmpfile[64];
        snprintf(tmpfile, sizeof(tmpfile), "/tmp/inf_req_%d", (int)getpid());
        int tf = open(tmpfile, O_WRONLY|O_CREAT|O_TRUNC, 0600);
        if (tf < 0 || !write_all(tf, req, len)) { close(tf); unlink(tmpfile); return false; }
        close(tf);
        int rf = open(tmpfile, O_RDONLY);
        if (rf < 0) return false;
        char buf[4096];
        ssize_t nr;
        while ((nr = read(rf, buf, sizeof(buf))) > 0) {
            if (!write_all(wfd, buf, (size_t)nr)) { close(rf); unlink(tmpfile); return false; }
        }
        close(rf);
        unlink(tmpfile);
        return true;
    }

    LocalTokenizer* local_tok_{nullptr};
public:
    SubprocessEngine() {}
    ~SubprocessEngine() { stop(); }

    bool start(const char* model) {
        std::lock_guard<std::mutex> g(lock);
        // Determine binary path
        const char* bin = "./server/inference_server";
        const char* bin9b = "./server/inference_server_9b";
        const char* bin_int4 = "./server/inference_server_int4";
        const char* bin_int4_batched = "./server/inference_server_int4_batched";
        const char* bin_llama = "./server/inference_server_llama";
        const char* bin_gemma = "./server/inference_server_gemma";
        struct { const char* key; const char* path; } bin_dirs[] = {
            {"gemma", bin_gemma}, {"9b", bin9b},
            {"batched", bin_int4_batched}, {"llama", bin_llama},
            {"qwen3", bin_int4}, {"int4", bin_int4},
        };
        for (auto& d : bin_dirs) { if (strstr(model, d.key)) { bin = d.path; break; } }

        int pin[2], pout[2];
        if(pipe(pin)==-1 || pipe(pout)==-1) return false;
        pid = fork();
        if(pid == 0) {
            close(pin[1]); close(pout[0]);
            dup2(pin[0], STDIN_FILENO); close(pin[0]);
            dup2(pout[1], STDOUT_FILENO); close(pout[1]);
            execl(bin, "inference_server", model, (char*)nullptr);
            _exit(1);
        }
        close(pin[0]); close(pout[1]);
        wfd = pin[1];
        from_f = fdopen(pout[0], "r");
        setvbuf(from_f, nullptr, _IONBF, 0);
        ready = true;
        is_gemma = strstr(model, "gemma") != nullptr;
        local_tok_ = new LocalTokenizer(model);
        fprintf(stderr, "SubprocessEngine: pid=%d model=%s bin=%s\n", pid, model, bin);
        return true;
    }

    bool generate(const std::string& prompt, int max_tok, float temp, int top_k,
                  float rep_pen, std::vector<uint32_t>& tokens, std::string& text, bool stream = false) {
        auto req_start = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> g(lock);
        if(!ready) {
            g_error_count.fetch_add(1);
            return false;
        }

        std::string ep = escape_json(prompt);

        char req[16384];
        int len = snprintf(req, sizeof(req),
            "{\"prompts\":[\"%s\"],\"max_tokens\":%d,\"temperature\":%g,\"top_k\":%d,\"repetition_penalty\":%.1f,\"stream\":%d}\n",
            ep.c_str(), max_tok, temp, top_k, rep_pen, stream ? 1 : 0);
        if (len < 0 || (size_t)len >= sizeof(req)) len = sizeof(req) - 1;

        if (!send_request(req, len)) return false;

        if (!stream) {
            // Non-streaming: read complete JSON response
            fd_set rfds;
            struct timeval tv;
            FD_ZERO(&rfds);
            FD_SET(fileno(from_f), &rfds);
            tv.tv_sec = 30;
            tv.tv_usec = 0;
            int sel = select(fileno(from_f)+1, &rfds, nullptr, nullptr, &tv);
            if(sel <= 0) return false;

            char line[16384];
            // Loop until we get a non-empty line starting with '{' (skip "Ready." etc.)
            do { if(!fgets(line, sizeof(line), from_f)) return false; } while(line[0] != '{');

            tokens.clear(); text.clear();
            // Try batch format "tokens":[[...]] (1.7B/8B) then single "tokens":[...] (9B)
            char* t = strstr(line, "\"tokens\":[[");
            if(t) {
                t += 10;
                while(*t && *t!=']') {
                    while(*t && (*t<'0' || *t>'9') && *t!='-') t++;
                    if(*t && ( (*t>='0' && *t<='9') || *t=='-')) {
                        long v = strtol(t, &t, 10);
                        tokens.push_back((uint32_t)v);
                    }
                }
            } else {
                t = strstr(line, "\"tokens\":[");
                if(t) {
                    t += 9;
                    while(*t && *t!=']') {
                        while(*t && (*t<'0' || *t>'9') && *t!='-') t++;
                        if(*t && ( (*t>='0' && *t<='9') || *t=='-')) {
                            long v = strtol(t, &t, 10);
                            tokens.push_back((uint32_t)v);
                        }
                    }
                }
            }
            // Try batch "text":["..."] then single "text":"..."
            // For single generate, decode tokens locally (skip for Gemma — uses SentencePiece)
            if(!tokens.empty() && local_tok_ && !is_gemma) {
                text = local_tok_->decode(tokens);
            } else {
                char* s = strstr(line, "\"text\":\"");
                if(s) { s += 8; char* e = strchr(s, '"'); if(e) text = std::string(s, e-s); }
            }
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - req_start).count();
            record_request(!tokens.empty(), duration);
            return !tokens.empty();
        } else {
            // Streaming: read SSE lines until [DONE] with timeout
            tokens.clear(); text.clear();
            char buf[4096];
            fd_set rfds;
            struct timeval tv;
            int fd = fileno(from_f);
            time_t start = time(nullptr);
            const int timeout_sec = 120;
            while(true) {
                FD_ZERO(&rfds);
                FD_SET(fd, &rfds);
                int elapsed = (int)(time(nullptr) - start);
                tv.tv_sec = std::max(1, timeout_sec - elapsed);
                tv.tv_usec = 0;
                int sel = select(fd + 1, &rfds, nullptr, nullptr, &tv);
                if(sel <= 0) return !tokens.empty();  // timeout or error
                if(!fgets(buf, sizeof(buf), from_f)) return !tokens.empty();
                if(strncmp(buf, "data: ", 6) == 0) {
                    if(strcmp(buf + 6, "[DONE]\n") == 0) break;
                    // Parse: {"token":123,"text":"abc"}
                    char* p = buf + 6;
                    char* tok_s = strstr(p, "\"token\":");
                    char* txt_s = strstr(p, "\"text\":\"");
                    if(tok_s) {
                        long v = strtol(tok_s + 8, &tok_s, 10);
                        tokens.push_back((uint32_t)v);
                    }
                    if(txt_s) {
                        char* e = strchr(txt_s + 9, '"');
                        if(e) text += std::string(txt_s + 9, e - (txt_s + 9));
                    }
                }
            }
            return !tokens.empty();
        }
    }

    bool generate_batch(const std::vector<std::string>& prompts_in, int max_tok, float temp, int top_k,
                  float rep_pen, std::vector<std::vector<uint32_t>>& all_tokens, std::vector<std::string>& all_text) {
        auto req_start = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> g(lock);
        if(!ready) {
            g_error_count.fetch_add(1);
            return false;
        }

        // Build batch JSON: {"prompts":["p1","p2",...],"max_tokens":N,...}
        std::string ep;
        for(size_t pi = 0; pi < prompts_in.size(); pi++) {
            if(pi > 0) ep += ",";
            ep += "\"" + escape_json(prompts_in[pi]) + "\"";
        }
        char req[32768];
        int len = snprintf(req, sizeof(req),
            "{\"prompts\":[%s],\"max_tokens\":%d,\"temperature\":%g,\"top_k\":%d,\"repetition_penalty\":%.1f,\"stream\":0}\n",
            ep.c_str(), max_tok, temp, top_k, rep_pen);
        if (len < 0 || (size_t)len >= sizeof(req)) len = sizeof(req) - 1;

        if (!send_request(req, len)) return false;

        // Read JSON response: skip non-JSON lines, get the actual response
        char line[65536];
        do { if(!fgets(line, sizeof(line), from_f)) return false; } while(line[0] != '{');

        // Parse tokens array
        char* p = strstr(line, "\"tokens\":[");
        if(!p) return false;
        p += 9;
        all_tokens.clear();
        while(*p && *p != ']') {
            while(*p && (*p < '0' || *p > '9') && *p != '[' && *p != '-') p++;
            if(*p == '[') {
                p++;
                std::vector<uint32_t> tok_seq;
                while(*p && *p != ']') {
                    while(*p && (*p < '0' || *p > '9') && *p != '-') p++;
                    if(*p && ((*p >= '0' && *p <= '9') || *p == '-')) {
                        long v = strtol(p, &p, 10);
                        tok_seq.push_back((uint32_t)v);
                    }
                }
                if(*p == ']') p++;
                all_tokens.push_back(tok_seq);
            } else { break; }
        }

        // Decode tokens locally using LocalTokenizer (skip for Gemma — SentencePiece)
        all_text.clear();
        for(const auto& ids : all_tokens) {
            if(local_tok_ && !is_gemma) {
                all_text.push_back(local_tok_->decode(ids));
            } else {
                all_text.push_back("");
            }
        }
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - req_start).count();
        record_request(!all_tokens.empty(), duration);
        return !all_tokens.empty();
    }

    void stop() {
        std::lock_guard<std::mutex> g(lock);
        if(pid > 0) { kill(pid, SIGKILL); waitpid(pid, nullptr, 0); }
        if(wfd >= 0) close(wfd);
        if(from_f) fclose(from_f);
        delete local_tok_; local_tok_ = nullptr;
        wfd = -1; pid = -1; ready = false;
    }

};

static SubprocessEngine g_engine;
// Subprocess restart: check if process is alive, restart if dead
static bool ensure_subprocess_alive(const char* model) {
    if (g_engine.ready) {
        int status;
        pid_t r = waitpid(g_engine.get_pid(), &status, WNOHANG);
        if (r == 0) return true;  // still alive
        // Process died — restart
        fprintf(stderr, "[RESTART] Subprocess died (status=%d). Restarting...\n", status);
    }
    g_engine.stop();
    return g_engine.start(model);
}

static std::string g_model_name;
static std::string g_model_raw; // original model arg for template selection

// Chat template per model architecture
// Llama 3: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
// Qwen3: <|im_start|>system\n...
static const char* get_chat_template(const char* model) {
    // g_model_name is set to display name ("1.7B", "8B") not model identifier.
    // The model string passed to start() contains the actual model name.
    if(strstr(model,"llama")) {
        return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
               "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
               "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    // Default Qwen3 template
    return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
}

std::string json_string_at(const std::string& body, const char* key) {
    std::string skey = "\"";
    skey += key;
    skey += "\":";
    size_t pos = body.find(skey);
    if(pos == std::string::npos) return "";
    pos += skey.size();
    while(pos < body.size() && (body[pos]==' ' || body[pos]=='\t')) pos++;
    if(pos >= body.size() || body[pos] != '"') return "";
    pos++;
    std::string val;
    while(pos < body.size()) {
        if(body[pos] == '\\' && pos+1 < body.size()) {
            pos++;
            if(body[pos] == 'n') val += '\n';
            else if(body[pos] == 'r') val += '\r';
            else if(body[pos] == 't') val += '\t';
            else if(body[pos] == '"') val += '"';
            else if(body[pos] == '\\') val += '\\';
            else val += body[pos];
        } else if(body[pos] == '"') {
            break;
        } else {
            val += body[pos];
        }
        pos++;
    }
    return val;
}

int json_int_at(const std::string& body, const char* key, int def) {
    std::string skey = "\"";
    skey += key;
    skey += "\":";
    size_t pos = body.find(skey);
    if(pos == std::string::npos) return def;
    pos += skey.size();
    while(pos < body.size() && (body[pos]<'0' || body[pos]>'9') && body[pos]!='-') pos++;
    if(pos >= body.size()) return def;
    return atoi(body.c_str() + pos);
}

float json_float_at(const std::string& body, const char* key, float def) {
    std::string skey = "\"";
    skey += key;
    skey += "\":";
    size_t pos = body.find(skey);
    if(pos == std::string::npos) return def;
    pos += skey.size();
    while(pos < body.size() && body[pos] != '-' && (body[pos]<'0' || body[pos]>'9') && body[pos]!='.') pos++;
    if(pos >= body.size()) return def;
    return atof(body.c_str() + pos);
}

std::string extract_chat_content(const std::string& body) {
    size_t pos = 0;
    while((pos = body.find("\"content\":", pos)) != std::string::npos) {
        pos += 10;
        while(pos < body.size() && (body[pos] == ' ' || body[pos] == '\t')) pos++;
        if(pos < body.size() && body[pos] == '"') {
            pos++;
            std::string val;
            while(pos < body.size()) {
                if(body[pos] == '\\' && pos+1 < body.size()) {
                    pos++;
                    if(body[pos] == 'n') val += '\n';
                    else if(body[pos] == 'r') val += '\r';
                    else if(body[pos] == 't') val += '\t';
                    else if(body[pos] == '"') val += '"';
                    else if(body[pos] == '\\') val += '\\';
                    else val += body[pos];
                } else if(body[pos] == '"') {
                    break;
                } else {
                    val += body[pos];
                }
                pos++;
            }
            return val;
        }
        pos++;
    }
    return "";
}

std::string escape_json_str(const std::string& s) {
    std::string r;
    for(size_t i=0;i<s.size();i++) {
        unsigned char c=s[i];
        if(c=='<') r+="\u003c";  // XSS guard
        else if(c=='>') r+="\u003e";
        else if(c=='{') r+="\u007b";
        else if(c=='}') r+="\u007d";
        else if(c=='"') r+="\\\"";
        else if(c=='\\') r+="\\\\";
        else if(c=='\n') r+="\\n";
        else if(c=='\r') r+="\\r";
        else if(c=='\t') r+="\\t";
        else if(c < 0x20 || c == 0x7f) r+=" ";  // control chars → space
        else if(c >= 0x80) {  // non-ASCII: escape as \uXXXX
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", c);
            r += buf;
        } else r+=c;
    }
    return r;
}

int main(int argc, char** argv) {
    int port = 8123;
    const char* model = "1.7b";
    for(int i=1;i<argc;i++) {
        if(strcmp(argv[i],"-p")==0 && i+1<argc) { port=atoi(argv[++i]); }
        else {
            int v = atoi(argv[i]);
            if(v > 0 && strspn(argv[i],"0123456789") == strlen(argv[i])) port = v;
            else model = argv[i];
        }
    }
    g_model_raw = model;
    if(strstr(model,"gemma")) g_model_name = "Gemma-12B";
    else if(strstr(model,"8b")) g_model_name = "8B";
    else if(strstr(model,"9b")) g_model_name = "9B";
    else if(strstr(model,"batched")) g_model_name = "8B";
    else g_model_name = "1.7B";

    fprintf(stderr, "Blackwell HTTP Server\n  Model: %s\n  Port: %d\n", model, port);

    if(!g_engine.start(model)) {
        fprintf(stderr, "FAIL: could not start inference_server\n"); return 1;
    }
    sleep(2);

    httplib::Server svr;
    svr.set_read_timeout(300);
    svr.set_write_timeout(300);
    svr.set_payload_max_length(1024 * 1024);  // 1MB

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        // Get GPU memory info
        int gpu_used_mb = 0, gpu_total_mb = 15849;
        FILE* smi = popen("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null", "r");
        if (smi) {
            setlocale(LC_NUMERIC, "C");
            if (fscanf(smi, "%d, %d", &gpu_used_mb, &gpu_total_mb) != 2) gpu_used_mb = 0;
            pclose(smi);
        }
        
        // Compute uptime
        auto now = std::chrono::steady_clock::now();
        auto uptime_sec = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time).count();
        
        // Compute avg latency
        uint64_t req_cnt = g_request_count.load();
        uint64_t err_cnt = g_error_count.load();
        uint64_t total_lat = g_total_latency_ms.load();
        double avg_lat = (req_cnt > 0) ? (double)total_lat / req_cnt : 0.0;
        
        char js[512];
        snprintf(js, sizeof(js),
            R"({"status":"ok","model":"blackwell-%s","gpu_used_mb":%d,"gpu_total_mb":%d,"uptime_sec":%ld,"requests":%lu,"errors":%lu,"avg_latency_ms":%.1f})",
            g_model_name.c_str(), gpu_used_mb, gpu_total_mb,
            uptime_sec, req_cnt, err_cnt, avg_lat);
        res.set_content(js, "application/json");
    });

    svr.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        uint64_t req_cnt = g_request_count.load();
        uint64_t err_cnt = g_error_count.load();
        uint64_t total_lat = g_total_latency_ms.load();
        double avg_lat = (req_cnt > 0) ? (double)total_lat / req_cnt : 0.0;
        auto now = std::chrono::steady_clock::now();
        auto uptime_sec = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time).count();
        
        char js[1024];
        snprintf(js, sizeof(js),
            "# HELP blackwell_requests_total Total requests\n"
            "# TYPE blackwell_requests_total counter\n"
            "blackwell_requests_total %lu\n"
            "# HELP blackwell_errors_total Total errors\n"
            "# TYPE blackwell_errors_total counter\n"
            "blackwell_errors_total %lu\n"
            "# HELP blackwell_latency_ms Average latency in ms\n"
            "# TYPE blackwell_latency_ms gauge\n"
            "blackwell_latency_ms %.1f\n"
            "# HELP blackwell_uptime_seconds Uptime in seconds\n"
            "# TYPE blackwell_uptime_seconds gauge\n"
            "blackwell_uptime_seconds %ld\n",
            req_cnt, err_cnt, avg_lat, uptime_sec);
        res.set_content(js, "text/plain; charset=utf-8");
    });

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        char js[256];
        snprintf(js, sizeof(js),
            R"({"object":"list","data":[{"id":"blackwell-%s","object":"model","created":0,"owned_by":"blackwell","root":"blackwell-%s"}]})",
            g_model_name.c_str(), g_model_name.c_str());
        res.set_content(js, "application/json");
    });

    svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        if (!g_rate_limiter.allow()) {
            res.status = 429;
            res.set_content(R"({"error":{"message":"Too many requests","type":"rate_limit_error"}})", "application/json");
            return;
        }
        if (!ensure_subprocess_alive(g_model_raw.c_str())) {
            res.status = 503;
            res.set_content(R"({"error":{"message":"Subprocess unavailable","type":"internal_error"}})", "application/json");
            return;
        }
        const std::string& body = req.body;
        std::string content = extract_chat_content(body);
        if(content.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"No content found","type":"invalid_request_error"}})", "application/json");
            return;
        }
        int max_tokens = json_int_at(body, "max_tokens", 30);
        if (max_tokens < 1) max_tokens = 1;
        if (max_tokens > 2048) max_tokens = 2048;
        float temp = json_float_at(body, "temperature", 0.0f);
        int top_k = json_int_at(body, "top_k", 0);
        float rep_pen = json_float_at(body, "repetition_penalty", 1.5f);

        char prompt_buf[65536];
        snprintf(prompt_buf, sizeof(prompt_buf), get_chat_template(g_model_raw.c_str()), content.c_str());
        std::string prompt = prompt_buf;

        std::vector<uint32_t> tokens; std::string text;
        if(!g_engine.generate(prompt, max_tokens, temp, top_k, rep_pen, tokens, text)) {
            res.status = 504;
            res.set_content(R"({"error":{"message":"Generation timeout or error","type":"internal_error"}})", "application/json");
            return;
        }

        std::ostringstream js;
        js << "{\"id\":\"chatcmpl-" << g_request_count.load() << "\",\"object\":\"chat.completion\",\"created\":" << time(nullptr) << ",\"model\":\"blackwell-" << g_model_name << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"";
        js << escape_json_str(text);
        js << "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":" << tokens.size() << ",\"total_tokens\":" << (tokens.size() + 1) << "}}";
        res.set_content(js.str(), "application/json");
    });

    svr.Post("/v1/completions", [](const httplib::Request& req, httplib::Response& res) {
        // Rate limit
        if (!g_rate_limiter.allow()) {
            res.status = 429;
            res.set_content(R"({"error":{"message":"Too many requests","type":"rate_limit_error"}})", "application/json");
            return;
        }
        // Ensure subprocess is alive
        if (!ensure_subprocess_alive(g_model_raw.c_str())) {
            res.status = 503;
            res.set_content(R"({"error":{"message":"Subprocess unavailable","type":"internal_error"}})", "application/json");
            return;
        }
        const std::string& body = req.body;
        std::string prompt = json_string_at(body, "prompt");
        if(prompt.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"No prompt found","type":"invalid_request_error"}})", "application/json");
            return;
        }
        int max_tokens = json_int_at(body, "max_tokens", 30);
        if (max_tokens < 1) max_tokens = 1;
        if (max_tokens > 2048) max_tokens = 2048;
        float temp = json_float_at(body, "temperature", 0.0f);
        int top_k = json_int_at(body, "top_k", 0);
        float rep_pen = json_float_at(body, "repetition_penalty", 1.5f);
        bool stream = json_int_at(body, "stream", 0) == 1;

        std::vector<uint32_t> tokens; std::string text;
        if(!g_engine.generate(prompt, max_tokens, temp, top_k, rep_pen, tokens, text, stream)) {
            res.status = 504;
            res.set_content(R"({"error":{"message":"Generation timeout or error","type":"internal_error"}})", "application/json");
            return;
        }

        std::ostringstream js;
        js << "{\"id\":\"cmpl-" << g_request_count.load() << "\",\"object\":\"text_completion\",\"created\":" << time(nullptr) << ",\"model\":\"blackwell-" << g_model_name << "\",\"choices\":[{\"text\":\"";
        js << escape_json_str(text);
        js << "\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":" << tokens.size() << ",\"total_tokens\":" << tokens.size() << "}}";
        res.set_content(js.str(), "application/json");
    });


    // Batch: POST {"prompts":["...","..."],"max_tokens":N} → 3-4× faster per token
    svr.Post("/v1/batch", [](const httplib::Request& req, httplib::Response& res) {
        if (!g_rate_limiter.allow()) {
            res.status = 429;
            res.set_content(R"({"error":{"message":"Too many requests","type":"rate_limit_error"}})", "application/json");
            return;
        }
        if (!ensure_subprocess_alive(g_model_raw.c_str())) {
            res.status = 503;
            res.set_content(R"({"error":{"message":"Subprocess unavailable","type":"internal_error"}})", "application/json");
            return;
        }
        std::vector<std::string> prompts;
        const char* bp = strstr(req.body.c_str(), "\"prompts\":");
        if(bp && (bp = strchr(bp, '['))) {
            bp++;
            while(*bp && *bp != ']') {
                while(*bp && *bp != '"') bp++;
                if(*bp == '"') { bp++; std::string t; while(*bp && *bp != '"') { if(*bp=='\\' && bp[1]) bp++; t+=*bp++; } if(*bp=='"') bp++; prompts.push_back(t); }
                while(*bp && *bp!='"' && *bp!=']') bp++;
            }
        }
        if(prompts.empty()) { res.status=400; res.set_content(R"({"error":{"message":"no prompts"}})", "application/json"); return; }
        if(prompts.size()>8) prompts.resize(8);
        int mt=json_int_at(req.body,"max_tokens",30);
        if(mt<1) mt=1; if(mt>2048) mt=2048;
        float tp=json_float_at(req.body,"temperature",0.0f);
        int tk=json_int_at(req.body,"top_k",0);
        float rp=json_float_at(req.body,"repetition_penalty",1.5f);
        std::vector<std::vector<uint32_t>> at; std::vector<std::string> ax;
        if(!g_engine.generate_batch(prompts,mt,tp,tk,rp,at,ax)) { res.status=504; res.set_content(R"({"error":{"message":"timeout"}})","application/json"); return; }
        std::ostringstream js; js<<"{\"batches\":[";
        for(size_t i=0;i<at.size();i++) { if(i)js<<","; js<<"{\"id\":\"b"<<i<<"\",\"choices\":[{\"text\":\""<<escape_json_str(ax[i])<<"\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":"<<at[i].size()<<"}}"; }
        js<<"]}"; res.set_content(js.str(),"application/json");
    });

    svr.Post("/v1/completions/stream", [](const httplib::Request& req, httplib::Response& res) {
        if (!g_rate_limiter.allow()) {
            res.status = 429;
            res.set_content(R"({"error":{"message":"Too many requests","type":"rate_limit_error"}})", "application/json");
            return;
        }
        if (!ensure_subprocess_alive(g_model_raw.c_str())) {
            res.status = 503;
            res.set_content(R"({"error":{"message":"Subprocess unavailable","type":"internal_error"}})", "application/json");
            return;
        }
        const std::string& body = req.body;
        std::string prompt = json_string_at(body, "prompt");
        if(prompt.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"No prompt found","type":"invalid_request_error"}})", "application/json");
            return;
        }
        int max_tokens = json_int_at(body, "max_tokens", 30);
        if (max_tokens < 1) max_tokens = 1;
        if (max_tokens > 2048) max_tokens = 2048;
        float temp = json_float_at(body, "temperature", 0.0f);
        int top_k = json_int_at(body, "top_k", 0);
        float rep_pen = json_float_at(body, "repetition_penalty", 1.5f);

        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");

        // True streaming: ContentProvider handles reading SSE tokens from subprocess
        res.set_content_provider(
            "text/event-stream",
            [=](size_t offset, httplib::DataSink &sink) -> bool {
                (void)offset;
                // Write request to subprocess stdin
                std::string ep;
                for(size_t i=0;i<prompt.size();i++) {
                    char c = prompt[i];
                    if(c=='\\') ep+="\\\\";
                    else if(c=='"') ep+="\\\"";
                    else if(c=='\n') ep+="\\n";
                    else if(c=='\r') ep+="\\r";
                    else if(c=='\t') ep+="\\t";
                    else ep+=c;
                }
                char req[16384];
                int len = snprintf(req, sizeof(req),
                    "{\"prompts\":[\"%s\"],\"max_tokens\":%d,\"temperature\":%g,\"top_k\":%d,\"repetition_penalty\":%.1f,\"stream\":1}\n",
                    ep.c_str(), max_tokens, temp, top_k, rep_pen);
                if (len < 0 || (size_t)len >= sizeof(req)) len = sizeof(req) - 1;
                char tmpfile[] = "/tmp/inf_req_XXXXXX";
                int tf = mkstemp(tmpfile);
                if(tf >= 0) {
                    write(tf, req, len);
                    close(tf);
                    int rf = open(tmpfile, O_RDONLY);
                    if(rf >= 0) {
                        char buf[4096];
                        ssize_t nr;
                        while((nr = read(rf, buf, sizeof(buf))) > 0) {
                            write(g_engine.get_write_fd(), buf, nr);
                        }
                        close(rf);
                    }
                    unlink(tmpfile);
                }

                // Read SSE tokens and stream them
                char buf[4096];
                fd_set rfds;
                struct timeval tv;
                int fd = fileno(g_engine.get_stream());
                time_t start = time(nullptr);
                const int timeout_sec = 120;
                while(true) {
                    FD_ZERO(&rfds);
                    FD_SET(fd, &rfds);
                    tv.tv_sec = std::max(1, timeout_sec - (int)(time(nullptr) - start));
                    tv.tv_usec = 0;
                    int sel = select(fd + 1, &rfds, nullptr, nullptr, &tv);
                    if(sel <= 0) break;
                    if(!fgets(buf, sizeof(buf), g_engine.get_stream())) break;
                    size_t blen = strlen(buf);
                    if(!sink.write(buf, blen)) break;
                    if(strncmp(buf, "data: ", 6) == 0 && strcmp(buf + 6, "[DONE]\n") == 0) break;
                }
                sink.done();
                return true;
            }
        );
    });

    fprintf(stderr, "Listening on port %d...\n", port);
    svr.listen("0.0.0.0", port);
    return 0;
}
// =====================================================================
// BPE Tokenizer (for local token decoding)
// =====================================================================
