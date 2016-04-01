/* This file is part of libLiFFT.
 *
 * libLiFFT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libLiFFT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libLiFFT.  If not, see <www.gnu.org/licenses/>.
 */
 
#pragma once

namespace tiffWriter {

    /**
     * Wraps an Allocator into a std-compliant allocator
     */
    template< typename T, class T_Allocator >
    struct AllocatorWrapper
    {
        using value_type = T;
        using Allocator = T_Allocator;
        using pointer = T*;
        using const_pointer = const T*;
        typedef size_t size_type;

        Allocator m_alloc;

        AllocatorWrapper(const Allocator& alloc = Allocator()): m_alloc(alloc) {}

        pointer
        allocate(size_type n, const void* = 0)
        {
            pointer p;
            m_alloc.malloc(p, n*sizeof(T));
            return p;
        }

        void
        deallocate(pointer p, size_type)
        {
            m_alloc.free(p);
        }
    };

    template< typename T, class T_Allocator >
    AllocatorWrapper<T, T_Allocator>
    wrapAllocator(const T_Allocator& alloc)
    {
        return AllocatorWrapper<T, T_Allocator>(alloc);
    }

}  // namespace tiffWriter
