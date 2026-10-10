// Minimal JSON reader for task files and the warm-mode protocol (objects, arrays, strings, numbers, true/false/null).
// Numbers keep their source text so integers above 2^53 and exact decimal values survive.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sf {

class Json {
 public:
  enum Type { Null, Bool, Number, String, Array, Object };

  static Json parse(const std::string& text) {
    size_t i = 0;
    Json v = value(text, i);
    skip(text, i);
    if (i != text.size()) throw std::runtime_error("json: trailing characters");
    return v;
  }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Null; }
  bool has(const std::string& key) const { return type_ == Object && obj_.count(key); }
  const Json& operator[](const std::string& key) const {
    if (type_ != Object) throw std::runtime_error("json: not an object (key " + key + ")");
    const auto it = obj_.find(key);
    if (it == obj_.end()) throw std::runtime_error("json: missing key " + key);
    return it->second;
  }
  const Json& operator[](size_t i) const {
    if (type_ != Array || i >= arr_.size()) throw std::runtime_error("json: bad index");
    return arr_[i];
  }
  size_t size() const { return type_ == Array ? arr_.size() : obj_.size(); }
  const std::vector<Json>& items() const { return arr_; }
  const std::map<std::string, Json>& members() const { return obj_; }

  const std::string& str() const {
    if (type_ != String) throw std::runtime_error("json: not a string");
    return str_;
  }
  double num() const {
    if (type_ != Number) throw std::runtime_error("json: not a number");
    return std::strtod(str_.c_str(), nullptr);
  }
  int64_t int64() const {
    if (type_ != Number || str_.find_first_of(".eE") != std::string::npos) throw std::runtime_error("json: not an integer");
    return std::strtoll(str_.c_str(), nullptr, 10);
  }
  bool boolean() const {
    if (type_ != Bool) throw std::runtime_error("json: not a boolean");
    return b_;
  }
  // value or a default when the key is absent
  double num_or(const std::string& key, double d) const { return has(key) ? (*this)[key].num() : d; }
  int64_t int_or(const std::string& key, int64_t d) const { return has(key) ? (*this)[key].int64() : d; }
  std::string str_or(const std::string& key, const std::string& d) const { return has(key) ? (*this)[key].str() : d; }
  bool bool_or(const std::string& key, bool d) const { return has(key) ? (*this)[key].boolean() : d; }

 private:
  Type type_ = Null;
  bool b_ = false;
  std::string str_;  // string value or number text
  std::vector<Json> arr_;
  std::map<std::string, Json> obj_;

  static void skip(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
  }
  static void expect(const std::string& s, size_t& i, char c) {
    skip(s, i);
    if (i >= s.size() || s[i] != c) throw std::runtime_error(std::string("json: expected ") + c);
    ++i;
  }
  static std::string string_at(const std::string& s, size_t& i) {
    expect(s, i, '"');
    std::string out;
    while (i < s.size() && s[i] != '"') {
      char c = s[i++];
      if (c == '\\') {
        if (i >= s.size()) break;
        const char e = s[i++];
        switch (e) {
          case 'n': c = '\n'; break;
          case 't': c = '\t'; break;
          case 'r': c = '\r'; break;
          case 'b': c = '\b'; break;
          case 'f': c = '\f'; break;
          case 'u': {  // ASCII only; task files are ASCII
            const long cp = std::strtol(s.substr(i, 4).c_str(), nullptr, 16);
            i += 4;
            if (cp > 0x7f) throw std::runtime_error("json: non-ASCII escape");
            c = static_cast<char>(cp);
            break;
          }
          default: c = e;
        }
      }
      out.push_back(c);
    }
    if (i >= s.size()) throw std::runtime_error("json: unterminated string");
    ++i;
    return out;
  }
  static Json value(const std::string& s, size_t& i) {
    skip(s, i);
    if (i >= s.size()) throw std::runtime_error("json: unexpected end");
    Json v;
    const char c = s[i];
    if (c == '{') {
      v.type_ = Object;
      ++i;
      skip(s, i);
      if (i < s.size() && s[i] == '}') return ++i, v;
      for (;;) {
        std::string key = string_at(s, i);
        expect(s, i, ':');
        v.obj_[key] = value(s, i);
        skip(s, i);
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        expect(s, i, '}');
        return v;
      }
    }
    if (c == '[') {
      v.type_ = Array;
      ++i;
      skip(s, i);
      if (i < s.size() && s[i] == ']') return ++i, v;
      for (;;) {
        v.arr_.push_back(value(s, i));
        skip(s, i);
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        expect(s, i, ']');
        return v;
      }
    }
    if (c == '"') {
      v.type_ = String;
      v.str_ = string_at(s, i);
      return v;
    }
    if (s.compare(i, 4, "true") == 0) return i += 4, v.type_ = Bool, v.b_ = true, v;
    if (s.compare(i, 5, "false") == 0) return i += 5, v.type_ = Bool, v;
    if (s.compare(i, 4, "null") == 0) return i += 4, v;
    const size_t j = s.find_first_not_of("+-0123456789.eE", i);
    if (j == i) throw std::runtime_error("json: unexpected character");
    v.type_ = Number;
    v.str_ = s.substr(i, j - i);
    i = j == std::string::npos ? s.size() : j;
    return v;
  }
};

// JSON string literal with escapes.
inline std::string json_quote(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\', out += c;
    else if (c == '\n') out += "\\n";
    else if (static_cast<unsigned char>(c) < 0x20) out += ' ';
    else out += c;
  }
  return out + "\"";
}

}  // namespace sf
