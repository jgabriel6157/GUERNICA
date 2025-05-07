#include "InputConfig.hxx"

InputConfig::InputConfig(const std::string &filename)
{
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line))
    {
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos)
            line = line.substr(0, comment_pos);

        std::istringstream iss(line);
        std::string key, eq, value;
        if (iss >> key >> eq >> value)
        {
            if (eq == "=")
                params[key] = value;
        }
    }
}
