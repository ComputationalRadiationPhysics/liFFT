#pragma once

namespace foobar {
namespace traits {

    /**
     * Evaluates to true type if the given type is an In-Place Indicator type
     * That is: Input data ptr should be uses as Output data ptr
     * and data format matches input data format except the type (complex or real)
     */
    template< class T >
    struct IsInplace: std::false_type{};

}  // namespace traits
}  // namespace foobar
