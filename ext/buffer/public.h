#ifndef PUBLIC_H
#define PUBLIC_H

#if !defined(PUBLIC)
#define PUBLIC
#elif defined(_WIN32)
#define PUBLIC __declspec(dllexport)
#endif

#endif