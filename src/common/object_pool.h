/* Copyright (c) 2017-2022 Hans-Kristian Arntzen
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <memory>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstdlib>

void *memalign_alloc(size_t boundary, size_t size);
void memalign_free(void *ptr);

template <typename T>
struct AlignedAllocation {
    static void* operator new(size_t size) {
        void* ret = memalign_alloc(alignof(T), size);
        if (!ret) throw std::bad_alloc();
        return ret;
    }

    static void* operator new[](size_t size) {
        void* ret = memalign_alloc(alignof(T), size);
        if (!ret) throw std::bad_alloc();
        return ret;
    }

    static void operator delete(void *ptr) {
        return memalign_free(ptr);
    }

    static void operator delete[](void *ptr) {
        return memalign_free(ptr);
    }
};

/**
 * Allocates objects of type T in batches of 64 * n where
 * n is the number of times the pool has grown. So the first
 * time it will allocate 64, then 128 objects etc.
 */
template<typename T>
class ObjectPool {
public:
    ObjectPool() {
        vacants.reserve(32);
    }

    template<typename... P>
    T* Allocate(P&&... p) {
#ifndef OBJECT_POOL_DEBUG
        if (vacants.empty()) {
            unsigned num_objects = 64u << memory.size();
            T *ptr = static_cast<T*>(memalign_alloc(std::max(64, alignof(T)),
                                                    num_objects * sizeof(T)));
            if (!ptr) {
                return nullptr;
            }

            for (unsigned i = 0; i < num_objects; i++) {
                vacants.push_back(&ptr[i]);
            }

            memory.emplace_back(ptr);
        }

        T *ptr = vacants.back();
        vacants.pop_back();
        new(ptr) T(std::forward<P>(p)...);
        return ptr;
#else
        return new T(std::forward<P>(p)...);
#endif
    }

    void Free(T *ptr) {
#ifndef OBJECT_POOL_DEBUG
        ptr->~T();
        vacants.push_back(ptr);
#else
        delete ptr;
#endif
    }

    void Clear() {
#ifndef OBJECT_POOL_DEBUG
        vacants.clear();
        memory.clear();
#endif
    }

protected:
#ifndef OBJECT_POOL_DEBUG
    std::vector<T*> vacants;

    struct MallocDeleter {
        void operator()(T *ptr) {
            memalign_free(ptr);
        }
    };

    std::vector<std::unique_ptr<T, MallocDeleter>> memory;
#endif
};

template<typename T>
class ThreadSafeObjectPool : private ObjectPool<T> {
public:
    template<typename... P>
    T* Allocate(P &&... p) {
        std::lock_guard<std::mutex> holder{lock};
        return ObjectPool<T>::Allocate(std::forward<P>(p)...);
    }

    void Free(T *ptr) {
#ifndef OBJECT_POOL_DEBUG
        ptr->~T();
        std::lock_guard<std::mutex> holder{lock};
        this->vacants.push_back(ptr);
#else
        delete ptr;
#endif
    }

    void Clear() {
        std::lock_guard<std::mutex> holder{lock};
        ObjectPool<T>::Clear();
    }

private:
    std::mutex lock;
};
