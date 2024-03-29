/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v1.4
 *
 * @brief Improved CUDA APIs
 * @details Advatages:                                                      <br>
 *   - **clear semantic**: input, then output (google style)
 *   - **type checking**:
 *      - input and output must have the same type T
 *      - const checking for inputs
 *      - device symbols must be references
 *   - **no byte object sizes**: the number of bytes is  determined by looking
 *       the parameter type T
 *   - **fast debugging**:
 *      - in case of error the macro provides the file name, the line, the
 *        name of the function where it is called, and the API name that fail
 *      - assertion to check null pointers and num_items == 0
 *      - assertion to check every CUDA API errors
 *      - additional info: cudaMalloc fail -> what is the available memory?
 *   - **direct argument passing** of constant values. E.g.                 <br>
 *       \code{.cu}
 *        cuMemcpyToSymbol(false, d_symbol); //d_symbol must be bool
 *       \endcode
 *   - much **less verbose**
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#pragma once

#include "Host/Basic.hpp"    //xlib::byte_t
#include "Host/Numeric.hpp"  //xlib::upper_approx
#include <cassert>           //assert
#include <cuda_runtime.h>    //cudaError_t
#include <utility>           //std::forward
#include <rmm/rmm.h>

#if defined(NEVER_DEFINED)
    #include "SafeFunctions_.cuh"
#endif

///@cond

#if !defined(NO_CHECK_CUDA_ERROR)
    #define CHECK_CUDA_ERROR                                                   \
        {                                                                      \
            cudaDeviceSynchronize();                                           \
            xlib::detail::getLastCudaError(__FILE__, __LINE__, __func__);      \
        }
#else
    #define CHECK_CUDA_ERROR
#endif

#define SAFE_CALL(function)                                                    \
    {                                                                          \
        xlib::detail::safe_call(function, __FILE__, __LINE__, __func__);       \
    }

int cuGetDeviceCount() noexcept;

void cuSetDevice(int device_index) noexcept;

int cuGetDevice() noexcept;

//==============================================================================

#define cuGetSymbolAddress(symbol, ptr)                                        \
    xlib::detail::cuGetSymbolAddressAux(__FILE__, __LINE__, __func__,          \
                                        symbol, ptr)                           \

#define cuMallocHost(...)                                                      \
    xlib::detail::cuMallocHostAux(__FILE__, __LINE__, __func__, __VA_ARGS__)   \

#define cuFreeHost(...)                                                        \
    xlib::detail::cuFreeHostAux(__FILE__, __LINE__, __func__, __VA_ARGS__)     \

//==============================================================================

namespace xlib {

namespace detail {

inline
void cudaErrorHandler(cudaError_t error, const char* error_message,
                      const char* file, int line,
                      const char* func_name) {
    if (cudaSuccess != error) {
        std::cerr << Color::FG_RED << "\nCUDA error\n" << Color::FG_DEFAULT
                  << Emph::SET_UNDERLINE << file
                  << Emph::SET_RESET  << "(" << line << ")"
                  << " [ "
                  << Color::FG_L_CYAN << func_name << Color::FG_DEFAULT
                  << " ] : " << error_message
                  << " -> " << cudaGetErrorString(error)
                  << "(" << static_cast<int>(error) << ")\n";
        if (error == cudaErrorMemoryAllocation) {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            std::cerr << "\nActual allocated memory: " << std::setprecision(1)
                      << std::fixed << (total - free) / xlib::MB << " MB\n";
        }
        std::cerr << std::endl;
        assert(false);                                                  //NOLINT
        std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));
        std::exit(EXIT_FAILURE);
    }
}

inline
void getLastCudaError(const char* file, int line, const char* func_name) {
    cudaErrorHandler(cudaGetLastError(), "", file, line, func_name);
}


inline
void safe_call(cudaError_t error, const char* file, int line,
                 const char* func_name) {
    cudaErrorHandler(error, "", file, line, func_name);
}

template<typename T>
void cuGetSymbolAddressAux(const char* file, int line, const char* func_name,
                           T& symbol, T*& ptr) noexcept {
    cudaErrorHandler(cudaGetSymbolAddress((void**)&ptr, symbol),
                     "cudaGetSymbolAddress", file, line, func_name);
}

template<typename T, int SIZE>
void cuGetSymbolAddressAux(const char* file, int line, const char* func_name,
                           T (&symbol)[SIZE], T*& ptr) noexcept {
    cudaErrorHandler(cudaGetSymbolAddress((void**)&ptr, symbol),
                     "cudaGetSymbolAddress", file, line, func_name);
}

//==============================================================================
////////////////
//  cuMalloc  //
////////////////

//it may be better to move this inside a cpp file (similar to xlib::detail::cudaErrorHandler) if there is an appropriate place.
#define RMM_ERROR_HANDLER(caller_name, callee_name, result) do {                                                   \
    std::cerr << xlib::Color::FG_RED << "\nRMM error\n" << xlib::Color::FG_DEFAULT          \
        << xlib::Emph::SET_UNDERLINE << __FILE__                                            \
        << xlib::Emph::SET_RESET  << "(" << __LINE__ << ")"                                 \
        << " [ "                                                                            \
        << xlib::Color::FG_L_CYAN << (caller_name) << xlib::Color::FG_DEFAULT \
        << " ] : " << callee_name                                                              \
        << " -> " << rmmGetErrorString(result)                                              \
        << "(" << static_cast<int>(result) << ")\n";                                        \
    assert(false);                                                                          \
    std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));                              \
    std::exit(EXIT_FAILURE);                                                                \
} while (0)

//------------------------------------------------------------------------------
template<typename T>
size_t byte_size(T* ptr, size_t num_items) noexcept {
    return num_items * sizeof(T);
}

template<typename T, typename... TArgs>
size_t byte_size(T* ptr, size_t num_items, TArgs... args) noexcept {
    return xlib::upper_approx<512>(num_items * sizeof(T)) + byte_size(args...);
}

template<typename T>
void set_ptr(xlib::byte_t* base_ptr, T*& ptr, size_t) noexcept {
    ptr = reinterpret_cast<T*>(base_ptr);
}

template<typename T, typename... TArgs>
void set_ptr(xlib::byte_t* base_ptr, T*& ptr, size_t num_items, TArgs... args)
             noexcept {
    ptr = reinterpret_cast<T*>(base_ptr);
    set_ptr(base_ptr + xlib::upper_approx<512>(num_items * sizeof(T)), args...);
}
/*
template<typename... TArgs>
void cuMallocAux(const char* file, int line, const char* func_name,
                 TArgs&&... args) noexcept {
    size_t num_bytes = byte_size(args...);
    assert(num_bytes > 0);
    xlib::byte_t* base_ptr;
    xlib::__cudaErrorHandler(cudaMalloc(&base_ptr, num_bytes), "cudaMalloc",
                             file, line, func_name);
    set_ptr(base_ptr, std::forward<TArgs>(args)...);
}*/

template<typename... TArgs>
void cuMallocHostAux(const char* file, int line, const char* func_name,
                     TArgs&&... args) noexcept {
    size_t num_bytes = byte_size(args...);
    assert(num_bytes > 0);
    xlib::byte_t* base_ptr;
    cudaErrorHandler(cudaMallocHost(&base_ptr, num_bytes),
                     "cudaMalloc", file, line, func_name);
    set_ptr(base_ptr, std::forward<TArgs>(args)...);
}

//------------------------------------------------------------------------------

template<typename T>
typename std::enable_if<std::is_pointer<T>::value>::type
cuFreeHostAux(const char* file, int line, const char* func_name, T ptr)
              noexcept {
    using   R = typename std::remove_cv<T>::type;
    auto ptr1 = const_cast<R>(ptr);
    cudaErrorHandler(cudaFreeHost(ptr1), "cudaFreeHost", file, line, func_name);
}

template<typename T, typename... TArgs>
typename std::enable_if<std::is_pointer<T>::value>::type
cuFreeHostAux(const char* file, int line, const char* func_name,
                   T ptr, TArgs... ptrs) noexcept {
    using   R = typename std::remove_cv<T>::type;
    auto ptr1 = const_cast<R>(ptr);
    cudaErrorHandler(cudaFreeHost(ptr1), "cudaFreeHost", file, line, func_name);
    cuFreeHostAux(file, line, func_name, ptrs...);
}

///@endcond

} // namespace detail
} // namespace xlib
