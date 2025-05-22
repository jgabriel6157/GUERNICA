#pragma once

#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>

class InputConfig
{
public:
    InputConfig(const std::string &filename);

    template<typename T>
    T Get(const std::string &key, T default_value) const
    {
        auto it = params.find(key);
        if (it != params.end())
        {
            std::istringstream iss(it->second);
            T val;
            iss >> std::boolalpha >> val;
            return val;
        }
        return default_value;
    }

private:
    std::unordered_map<std::string, std::string> params;
};