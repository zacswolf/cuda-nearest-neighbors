// exceptions.h
#pragma once

class NotImplementedException : public std::logic_error
{
public:
	NotImplementedException(string functionName) : std::logic_error(functionName + " not yet implemented") { };
};
