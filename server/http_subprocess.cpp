#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#define CPPHTTPLIB_ZLIB_SUPPORT 0
#include "blackwell/httplib.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <mutex>
#include <sys/wait.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Blackwell HTTP Server
// Uses temp-file IPC: writes request JSON to /tmp/inf_{pid}.req,
// inference_server reads from that file instead of stdin.

#include "blackwell/bpe_tokenizer.h"

class LocalTokenizer {
    blackwell::BpeTokenizer tok_;
    bool ok_{false};
public:
    LocalTokenizer(const char* weight_dir) {
        std::string td = "./tokenizer_data.bin";
        ok_ = tok_.load(td.c_str()) == 0;
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
    std::mutex lock;
    bool ready{false};
    LocalTokenizer* local_tok_{nullptr};
public:
    SubprocessEngine() {}
    ~SubprocessEngine() { stop(); }

    bool start(const char* model) {
        std::lock_guard<std::mutex> g(lock);
        // Determine binary path
        const char* bin = "./server/inference_server";
        std::string bin9b = "./server/inference_server_9b";
        std::string bin_int4 = "./server/inference_server_int4";
        if(strstr(model,"9b")) bin = bin9b.c_str();
        else if(strstr(model,"int4")) bin = bin_int4.c_str();

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
        local_tok_ = new LocalTokenizer(model);
        fprintf(stderr, "SubprocessEngine: pid=%d model=%s bin=%s\n", pid, model, bin);
        return true;
    }

    bool generate(const std::string& prompt, int max_tok, float temp, int top_k,
                  std::vector<uint32_t>& tokens, std::string& text, bool stream = false) {
        std::lock_guard<std::mutex> g(lock);
        if(!ready) return false;

        std::string ep;
        for(size_t i=0;i<prompt.size();i++) {
            char c = prompt[i];
            if(c=='\\') { ep+="\\\\"; }
            else if(c=='"') { ep+="\\\""; }
            else if(c=='\n') { ep+="\\n"; }
            else if(c=='\r') { ep+="\\r"; }
            else if(c=='\t') { ep+="\\t"; }
            else ep+=c;
        }

        char req[16384];
        int len = snprintf(req, sizeof(req),
            "{\"prompts\":[\"%s\"],\"max_tokens\":%d,\"temperature\":%g,\"top_k\":%d,\"stream\":%d}\n",
            ep.c_str(), max_tok, temp, top_k, stream ? 1 : 0);

        // Write to temp file, cat into subprocess stdin
        char tmpfile[64];
        snprintf(tmpfile, sizeof(tmpfile), "/tmp/inf_req_%d", getpid());
        int tf = open(tmpfile, O_WRONLY|O_CREAT|O_TRUNC, 0600);
        if(tf < 0) return false;
        ssize_t written = 0;
        while(written < len) {
            ssize_t n = write(tf, req + written, len - written);
            if(n < 0) { close(tf); return false; }
            written += n;
        }
        close(tf);

        // Write file content to subprocess stdin
        int rf = open(tmpfile, O_RDONLY);
        if(rf < 0) return false;
        char buf[4096];
        ssize_t nr;
        while((nr = read(rf, buf, sizeof(buf))) > 0) {
            ssize_t n = write(wfd, buf, nr);
            if(n < 0) { close(rf); return false; }
        }
        close(rf);
        unlink(tmpfile);

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
            // For single generate, decode tokens locally
            if(!tokens.empty() && local_tok_) {
                text = local_tok_->decode(tokens);
            } else {
                char* s = strstr(line, "\"text\":\"");
                if(s) { s += 8; char* e = strchr(s, '"'); if(e) text = std::string(s, e-s); }
            }
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
                  std::vector<std::vector<uint32_t>>& all_tokens, std::vector<std::string>& all_text) {
        std::lock_guard<std::mutex> g(lock);
        if(!ready) return false;

        // Build batch JSON: {"prompts":["p1","p2",...],"max_tokens":N,...}
        std::string ep;
        for(size_t pi = 0; pi < prompts_in.size(); pi++) {
            if(pi > 0) ep += ",";
            ep += "\"";
            for(size_t i = 0; i < prompts_in[pi].size(); i++) {
                char c = prompts_in[pi][i];
                if(c=='\\') { ep += "\\\\"; }
                else if(c=='"') { ep += "\\\""; }
                else if(c=='\n') { ep += "\\n"; }
                else if(c=='\r') { ep += "\\r"; }
                else if(c=='\t') { ep += "\\t"; }
                else ep += c;
            }
            ep += "\"";
        }
        char req[32768];
        int len = snprintf(req, sizeof(req),
            "{\"prompts\":[%s],\"max_tokens\":%d,\"temperature\":%g,\"top_k\":%d,\"stream\":0}\n",
            ep.c_str(), max_tok, temp, top_k);

        char tmpfile[64];
        snprintf(tmpfile, sizeof(tmpfile), "/tmp/inf_req_%d", (int)getpid());
        int tf = open(tmpfile, O_WRONLY|O_CREAT|O_TRUNC, 0600);
        if(tf < 0) return false;
        ssize_t written = 0;
        while(written < len) {
            ssize_t n = write(tf, req + written, len - written);
            if(n < 0) { close(tf); return false; }
            written += n;
        }
        close(tf);

        int rf = open(tmpfile, O_RDONLY);
        if(rf < 0) return false;
        char buf[4096];
        ssize_t nr;
        int wfd_local = wfd;
        while((nr = read(rf, buf, sizeof(buf))) > 0) {
            ssize_t n = write(wfd_local, buf, nr);
            if(n < 0) { close(rf); return false; }
        }
        close(rf);
        unlink(tmpfile);

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

        // Decode tokens locally using LocalTokenizer
        all_text.clear();
        for(const auto& ids : all_tokens) {
            all_text.push_back(local_tok_ ? local_tok_->decode(ids) : "");
        }
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
static std::string g_model_name;

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
    if(strstr(model,"8b")) g_model_name = "8B";
    else if(strstr(model,"9b")) g_model_name = "9B";
    else g_model_name = "1.7B";

    fprintf(stderr, "Blackwell HTTP Server\n  Model: %s\n  Port: %d\n", model, port);

    if(!g_engine.start(model)) {
        fprintf(stderr, "FAIL: could not start inference_server\n"); return 1;
    }
    sleep(2);

    httplib::Server svr;
    svr.set_read_timeout(300);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        char js[256];
        snprintf(js, sizeof(js),
            R"({"object":"list","data":[{"id":"blackwell-%s","object":"model","created":0,"owned_by":"blackwell","root":"blackwell-%s"}]})",
            g_model_name.c_str(), g_model_name.c_str());
        res.set_content(js, "application/json");
    });

    svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        const std::string& body = req.body;
        std::string content = extract_chat_content(body);
        if(content.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"No content found","type":"invalid_request_error"}})", "application/json");
            return;
        }
        int max_tokens = json_int_at(body, "max_tokens", 30);
        float temp = json_float_at(body, "temperature", 0.0f);
        int top_k = json_int_at(body, "top_k", 0);

        std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
        prompt += content;
        prompt += "<|im_end|>\n<|im_start|>assistant\n";

        std::vector<uint32_t> tokens; std::string text;
        if(!g_engine.generate(prompt, max_tokens, temp, top_k, tokens, text)) {
            res.status = 504;
            res.set_content(R"({"error":{"message":"Generation timeout or error","type":"internal_error"}})", "application/json");
            return;
        }

        std::ostringstream js;
        js << "{\"id\":\"chatcmpl-0\",\"object\":\"chat.completion\",\"created\":0,\"model\":\"blackwell-" << g_model_name << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"";
        js << escape_json_str(text);
        js << "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":" << tokens.size() << ",\"total_tokens\":" << tokens.size() << "}}";
        res.set_content(js.str(), "application/json");
    });

    svr.Post("/v1/completions", [](const httplib::Request& req, httplib::Response& res) {
        const std::string& body = req.body;
        std::string prompt = json_string_at(body, "prompt");
        if(prompt.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"No prompt found","type":"invalid_request_error"}})", "application/json");
            return;
        }
        int max_tokens = json_int_at(body, "max_tokens", 30);
        float temp = json_float_at(body, "temperature", 0.0f);
        int top_k = json_int_at(body, "top_k", 0);
        bool stream = json_int_at(body, "stream", 0) == 1;

        std::vector<uint32_t> tokens; std::string text;
        if(!g_engine.generate(prompt, max_tokens, temp, top_k, tokens, text, stream)) {
            res.status = 504;
            res.set_content(R"({"error":{"message":"Generation timeout or error","type":"internal_error"}})", "application/json");
            return;
        }

        std::ostringstream js;
        js << "{\"id\":\"cmpl-0\",\"object\":\"text_completion\",\"created\":0,\"model\":\"blackwell-" << g_model_name << "\",\"choices\":[{\"text\":\"";
        js << escape_json_str(text);
        js << "\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":" << tokens.size() << ",\"total_tokens\":" << tokens.size() << "}}";
        res.set_content(js.str(), "application/json");
    });


    // Batch: POST {"prompts":["...","..."],"max_tokens":N} → 3-4× faster per token
    svr.Post("/v1/batch", [](const httplib::Request& req, httplib::Response& res) {
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
        float tp=json_float_at(req.body,"temperature",0.0f);
        int tk=json_int_at(req.body,"top_k",0);
        std::vector<std::vector<uint32_t>> at; std::vector<std::string> ax;
        if(!g_engine.generate_batch(prompts,mt,tp,tk,at,ax)) { res.status=504; res.set_content(R"({"error":{"message":"timeout"}})","application/json"); return; }
        std::ostringstream js; js<<"{\"batches\":[";
        for(size_t i=0;i<at.size();i++) { if(i)js<<","; js<<"{\"id\":\"b"<<i<<"\",\"choices\":[{\"text\":\""<<escape_json_str(ax[i])<<"\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":"<<at[i].size()<<"}}"; }
        js<<"]}"; res.set_content(js.str(),"application/json");
    });

    svr.Post("/v1/completions/stream", [](const httplib::Request& req, httplib::Response& res) {
        const std::string& body = req.body;
        std::string prompt = json_string_at(body, "prompt");
        if(prompt.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"No prompt found","type":"invalid_request_error"}})", "application/json");
            return;
        }
        int max_tokens = json_int_at(body, "max_tokens", 30);
        float temp = json_float_at(body, "temperature", 0.0f);
        int top_k = json_int_at(body, "top_k", 0);

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
                    "{\"prompts\":[\"%s\"],\"max_tokens\":%d,\"temperature\":%g,\"top_k\":%d,\"stream\":1}\n",
                    ep.c_str(), max_tokens, temp, top_k);
                char tmpfile[64];
                snprintf(tmpfile, sizeof(tmpfile), "/tmp/inf_req_%d", (int)getpid());
                int tf = open(tmpfile, O_WRONLY|O_CREAT|O_TRUNC, 0600);
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
