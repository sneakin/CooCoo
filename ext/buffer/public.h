#ifndef PUBLIC_H
#define PUBLIC_H

#if !defined(PUBLIC)
#  if defined(_WIN32)
#    define PUBLIC __declspec(dllexport)
#  else
#    define PUBLIC
#  endif
#endif

#endif
