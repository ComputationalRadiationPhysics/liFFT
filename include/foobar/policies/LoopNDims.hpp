#pragma once

namespace foobar {
namespace policies {

    /**
     * Loops over N dimensions (last index varying fastest) and calls Functor(index, args...)
     */
    template< unsigned T_numDims >
    struct LoopNDims;

    template<>
    struct LoopNDims<1>
    {
        template<
            class T_Index,
            class T_Extents,
            typename T_Functor,
            typename... T_Args
        >
        static void loop(T_Index&& idx, const T_Extents& extents, T_Functor&& f, T_Args&& ... args)
        {
            for(idx[0]=0; idx[0]<extents[0]; ++idx[0])
                std::forward<T_Functor>(f)(idx, std::forward<T_Args>(args)...);
        }
    };

    template<>
    struct LoopNDims<2>
    {
        template<
            class T_Index,
            class T_Extents,
            typename T_Functor,
            typename... T_Args
        >
        static void loop(T_Index&& idx, const T_Extents& extents, T_Functor&& f, T_Args&& ... args)
        {
            for(idx[0]=0; idx[0]<extents[0]; ++idx[0])
                for(idx[1]=0; idx[1]<extents[1]; ++idx[1])
                    std::forward<T_Functor>(f)(idx, std::forward<T_Args>(args)...);
        }
    };

    template<>
    struct LoopNDims<3>
    {
        template<
            class T_Index,
            class T_Extents,
            typename T_Functor,
            typename... T_Args
        >
        static void loop(T_Index&& idx, const T_Extents& extents, T_Functor&& f, T_Args&& ... args)
        {
            for(idx[0]=0; idx[0]<extents[0]; ++idx[0])
                for(idx[1]=0; idx[1]<extents[1]; ++idx[1])
                    for(idx[2]=0; idx[2]<extents[2]; ++idx[2])
                        std::forward<T_Functor>(f)(idx, std::forward<T_Args>(args)...);
        }
    };

}  // namespace policies
}  // namespace foobar
