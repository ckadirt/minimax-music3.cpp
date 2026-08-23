// Adapted for minimax-music3.cpp from LeVo2.cpp's strict request parser.
// Modified file. Licensed under Apache-2.0.

#include "minimax-request.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace minimax::request_io {
namespace {

constexpr std::size_t max_request_bytes = 1024U * 1024U;

[[noreturn]] void fail(const std::string & message) {
    throw std::invalid_argument("MiniMax request: " + message);
}

bool valid_utf8(const std::string & source);

struct value {
    enum class kind { null_value, boolean, number, string, array, object } type = kind::null_value;
    double number = 0.0;
    std::string number_text;
    std::string string;
    std::vector<value> array;
    std::map<std::string, value> object;
};

class reader final {
public:
    explicit reader(const std::string & input) : input_(input) {}

    value parse() {
        skip();
        value result = parse_value();
        skip();
        if (position_ != input_.size()) fail("trailing bytes after JSON value");
        return result;
    }

private:
    void skip() {
        while (position_ != input_.size() &&
               (input_[position_] == ' ' || input_[position_] == '\n' ||
                input_[position_] == '\r' || input_[position_] == '\t')) ++position_;
    }

    char take() {
        if (position_ == input_.size()) fail("unexpected end of JSON");
        return input_[position_++];
    }

    void expect(char expected) {
        skip();
        if (take() != expected) fail(std::string("expected '") + expected + "'");
    }

    static unsigned hex(char c) {
        if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
        fail("invalid hexadecimal escape");
    }

    static void append_utf8(std::string & output, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) output.push_back(static_cast<char>(codepoint));
        else if (codepoint <= 0x7ffU) {
            output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0xffffU) {
            output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0x10ffffU) {
            output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else fail("Unicode escape is outside range");
    }

    std::uint32_t unicode_escape() {
        std::uint32_t result = 0;
        for (unsigned index = 0; index != 4; ++index) result = (result << 4U) | hex(take());
        return result;
    }

    std::string parse_string() {
        if (take() != '"') fail("expected string");
        std::string result;
        for (;;) {
            const unsigned char current = static_cast<unsigned char>(take());
            if (current == '"') {
                if (!valid_utf8(result)) fail("invalid UTF-8 in JSON string");
                return result;
            }
            if (current < 0x20U) fail("control byte in JSON string");
            if (current != '\\') {
                result.push_back(static_cast<char>(current));
                continue;
            }
            switch (take()) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    std::uint32_t codepoint = unicode_escape();
                    if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                        if (take() != '\\' || take() != 'u') fail("unpaired high surrogate");
                        const std::uint32_t low = unicode_escape();
                        if (low < 0xdc00U || low > 0xdfffU) fail("invalid low surrogate");
                        codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) + low - 0xdc00U;
                    } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
                        fail("unpaired low surrogate");
                    }
                    append_utf8(result, codepoint);
                    break;
                }
                default: fail("invalid JSON string escape");
            }
        }
    }

    value parse_object() {
        value result;
        result.type = value::kind::object;
        expect('{');
        skip();
        if (position_ != input_.size() && input_[position_] == '}') {
            ++position_;
            return result;
        }
        for (;;) {
            skip();
            const std::string key = parse_string();
            expect(':');
            if (!result.object.emplace(key, parse_value()).second) {
                fail("duplicate JSON object key '" + key + "'");
            }
            skip();
            const char separator = take();
            if (separator == '}') return result;
            if (separator != ',') fail("expected ',' or '}' in JSON object");
        }
    }

    value parse_array() {
        value result;
        result.type = value::kind::array;
        expect('[');
        skip();
        if (position_ != input_.size() && input_[position_] == ']') {
            ++position_;
            return result;
        }
        for (;;) {
            result.array.push_back(parse_value());
            skip();
            const char separator = take();
            if (separator == ']') return result;
            if (separator != ',') fail("expected ',' or ']' in JSON array");
        }
    }

    value parse_number() {
        const std::size_t begin = position_;
        if (input_[position_] == '-') ++position_;
        if (position_ == input_.size()) fail("invalid JSON number");
        if (input_[position_] == '0') {
            ++position_;
            if (position_ != input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
                fail("leading zero in JSON number");
            }
        } else if (input_[position_] >= '1' && input_[position_] <= '9') {
            while (position_ != input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
        } else fail("invalid JSON number");
        if (position_ != input_.size() && input_[position_] == '.') {
            const std::size_t fraction = ++position_;
            while (position_ != input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (position_ == fraction) fail("JSON fraction needs a digit");
        }
        if (position_ != input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ != input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            const std::size_t exponent = position_;
            while (position_ != input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (position_ == exponent) fail("JSON exponent needs a digit");
        }
        const char * first = input_.c_str() + begin;
        char * last = nullptr;
        const double parsed = std::strtod(first, &last);
        if (last != input_.c_str() + position_ || !std::isfinite(parsed)) fail("invalid JSON number");
        value result;
        result.type = value::kind::number;
        result.number = parsed;
        result.number_text.assign(first, static_cast<std::size_t>(last - first));
        return result;
    }

    value parse_value() {
        skip();
        if (position_ == input_.size()) fail("unexpected end of JSON value");
        const char current = input_[position_];
        if (current == '{') return parse_object();
        if (current == '[') return parse_array();
        if (current == '"') {
            value result;
            result.type = value::kind::string;
            result.string = parse_string();
            return result;
        }
        if (input_.compare(position_, 4, "true") == 0) {
            position_ += 4;
            value result;
            result.type = value::kind::boolean;
            return result;
        }
        if (input_.compare(position_, 5, "false") == 0) {
            position_ += 5;
            value result;
            result.type = value::kind::boolean;
            return result;
        }
        if (input_.compare(position_, 4, "null") == 0) {
            position_ += 4;
            return {};
        }
        if (current == '-' || (current >= '0' && current <= '9')) return parse_number();
        fail("invalid JSON value");
    }

    const std::string & input_;
    std::size_t position_ = 0;
};

const value * lookup(const value & object, const char * key) {
    const auto found = object.object.find(key);
    return found == object.object.end() ? nullptr : &found->second;
}

const value & required(const value & object, const char * key, value::kind type) {
    const value * result = lookup(object, key);
    if (result == nullptr || result->type != type) {
        fail(std::string("required field '") + key + "' has the wrong type");
    }
    return *result;
}

void reject_unknown(const value & object, const std::vector<std::string> & allowed, const char * scope) {
    for (const auto & item : object.object) {
        bool accepted = false;
        for (const auto & name : allowed) if (item.first == name) accepted = true;
        if (!accepted) fail(std::string("unknown ") + scope + " field '" + item.first + "'");
    }
}

std::uint64_t exact_u64(const value & source, const char * label) {
    if (source.type != value::kind::number || source.number_text.empty() ||
        source.number_text.front() == '-' || source.number_text.find_first_of(".eE") != std::string::npos) {
        fail(std::string("field '") + label + "' must be an unsigned integer");
    }
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(source.number_text.data(),
                                        source.number_text.data() + source.number_text.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != source.number_text.data() + source.number_text.size()) {
        fail(std::string("field '") + label + "' must be an unsigned integer");
    }
    return result;
}

std::size_t exact_size(const value & source, const char * label) {
    const std::uint64_t result = exact_u64(source, label);
    if (result > std::numeric_limits<std::size_t>::max()) fail(std::string("field '") + label + "' is too large");
    return static_cast<std::size_t>(result);
}

float finite_float(const value & source, const char * label) {
    if (source.type != value::kind::number || !std::isfinite(source.number) ||
        source.number < -std::numeric_limits<float>::max() ||
        source.number > std::numeric_limits<float>::max()) {
        fail(std::string("field '") + label + "' must be a finite float");
    }
    return static_cast<float>(source.number);
}

bool valid_utf8(const std::string & source) {
    for (std::size_t index = 0; index < source.size();) {
        const unsigned char first = static_cast<unsigned char>(source[index++]);
        if (first < 0x80U) continue;
        unsigned count = 0;
        std::uint32_t codepoint = 0;
        if (first >= 0xc2U && first <= 0xdfU) { count = 1; codepoint = first & 0x1fU; }
        else if (first >= 0xe0U && first <= 0xefU) { count = 2; codepoint = first & 0x0fU; }
        else if (first >= 0xf0U && first <= 0xf4U) { count = 3; codepoint = first & 0x07U; }
        else return false;
        if (source.size() - index < count) return false;
        for (unsigned offset = 0; offset != count; ++offset) {
            const unsigned char current = static_cast<unsigned char>(source[index++]);
            if ((current & 0xc0U) != 0x80U) return false;
            codepoint = (codepoint << 6U) | (current & 0x3fU);
        }
        if ((count == 1 && codepoint < 0x80U) || (count == 2 && codepoint < 0x800U) ||
            (count == 3 && codepoint < 0x10000U) ||
            (codepoint >= 0xd800U && codepoint <= 0xdfffU) || codepoint > 0x10ffffU) return false;
    }
    return true;
}

std::string escape(const std::string & source) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char current : source) {
        switch (current) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (current < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(current) << std::dec << std::setfill(' ');
                } else output << static_cast<char>(current);
        }
    }
    output << '"';
    return output.str();
}

std::string number(double source) {
    if (!std::isfinite(source)) fail("cannot serialize a non-finite number");
    std::ostringstream output;
    output << std::setprecision(17) << source;
    return output.str();
}

} // namespace

void validate(const generation_request & request) {
    if (!valid_utf8(request.lyrics) || !valid_utf8(request.description)) {
        fail("lyrics and description must be valid UTF-8");
    }
    if (request.lyrics.empty()) fail("lyrics must not be empty");
    if (request.description.empty()) fail("description must not be empty");
    if (!std::isfinite(request.duration_seconds) || request.duration_seconds <= 0.0 ||
        request.duration_seconds > 300.0) fail("duration_seconds must be in (0, 300]");
    if (!std::isfinite(request.ar_cfg_scale) || request.ar_cfg_scale < 0.0F || request.ar_cfg_scale > 100.0F) {
        fail("cfg_scale must be in [0, 100]");
    }
    if (request.top_k == 0 || request.top_k > 16384) fail("sampling.top_k must be in [1, 16384]");
    if (request.euler_steps == 0 || request.euler_steps > 1000) fail("flow.euler_steps must be in [1, 1000]");
    if (!std::isfinite(request.flow_cfg_scale) || request.flow_cfg_scale < 0.0F ||
        request.flow_cfg_scale > 100.0F) fail("flow.cfg_scale must be in [0, 100]");
    if (request.output_sample_rate != 44100 && request.output_sample_rate != 32000) {
        fail("output_sample_rate must be 44100 or 32000");
    }
}

generation_request parse(const std::string & json) {
    if (json.size() > max_request_bytes) fail("JSON exceeds the 1 MiB limit");
    const value root = reader(json).parse();
    if (root.type != value::kind::object) fail("request must be a JSON object");
    reject_unknown(root, {"lyrics", "description", "duration_seconds", "seed", "cfg_scale",
                          "sampling", "flow", "output_sample_rate"}, "request");

    generation_request result;
    result.lyrics = required(root, "lyrics", value::kind::string).string;
    result.description = required(root, "description", value::kind::string).string;
    const value & duration = required(root, "duration_seconds", value::kind::number);
    result.duration_seconds = duration.number;
    if (const value * item = lookup(root, "seed")) result.seed = exact_u64(*item, "seed");
    if (const value * item = lookup(root, "cfg_scale")) result.ar_cfg_scale = finite_float(*item, "cfg_scale");
    if (const value * item = lookup(root, "output_sample_rate")) {
        const std::uint64_t rate = exact_u64(*item, "output_sample_rate");
        if (rate > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) fail("output_sample_rate is too large");
        result.output_sample_rate = static_cast<int>(rate);
    }

    if (const value * sampling = lookup(root, "sampling")) {
        if (sampling->type != value::kind::object) fail("sampling must be an object");
        reject_unknown(*sampling, {"top_k"}, "sampling");
        if (const value * item = lookup(*sampling, "top_k")) result.top_k = exact_size(*item, "sampling.top_k");
    }
    if (const value * flow = lookup(root, "flow")) {
        if (flow->type != value::kind::object) fail("flow must be an object");
        reject_unknown(*flow, {"seed", "euler_steps", "cfg_scale"}, "flow");
        if (const value * item = lookup(*flow, "seed")) {
            result.flow_seed = exact_u64(*item, "flow.seed");
            result.flow_seed_present = true;
        }
        if (const value * item = lookup(*flow, "euler_steps")) {
            result.euler_steps = exact_size(*item, "flow.euler_steps");
        }
        if (const value * item = lookup(*flow, "cfg_scale")) {
            result.flow_cfg_scale = finite_float(*item, "flow.cfg_scale");
        }
    }
    validate(result);
    return result;
}

std::string serialize(const generation_request & request) {
    validate(request);
    std::ostringstream output;
    output << "{\"lyrics\":" << escape(request.lyrics)
           << ",\"description\":" << escape(request.description)
           << ",\"duration_seconds\":" << number(request.duration_seconds)
           << ",\"seed\":" << request.seed
           << ",\"cfg_scale\":" << number(request.ar_cfg_scale)
           << ",\"sampling\":{\"top_k\":" << request.top_k << "}"
           << ",\"flow\":{";
    if (request.flow_seed_present) output << "\"seed\":" << request.flow_seed << ',';
    output << "\"euler_steps\":" << request.euler_steps
           << ",\"cfg_scale\":" << number(request.flow_cfg_scale) << "}"
           << ",\"output_sample_rate\":" << request.output_sample_rate << '}';
    return output.str();
}

} // namespace minimax::request_io
