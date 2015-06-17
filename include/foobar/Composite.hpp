#pragma once

namespace foobar {

    template < class... T_Funcs >
    struct Composite;

    template< class T_Func >
    struct Composite< T_Func >: T_Func{};

    template< class T_Func, class... T_Funcs >
    struct Composite< T_Func, T_Funcs... >{

        T_Func f1;
        Composite< T_Funcs... > f2;

        template< typename... T >
        auto
        operator()(T&&... args)
        -> decltype( f1( f2( std::forward<T>(args)... ) ) )
        {
            return f1( f2( std::forward<T>(args)... ) );
        }
    };


}  // namespace foobar
