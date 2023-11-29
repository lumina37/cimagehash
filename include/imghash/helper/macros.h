#pragma once

#ifdef _MSC_VER
#define IGH_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#define IGH_FORCEINLINE __inline__ __attribute__((always_inline))
#else
#define IGH_FORCEINLINE inline
#endif
