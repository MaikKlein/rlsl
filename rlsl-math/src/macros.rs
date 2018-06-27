macro_rules! vec_impl_op {
    ($name: ident {$($fields:ident),*}, $trait: ident, $fn: ident, $op: tt) => {
        impl<T: Float> ::std::ops::$trait for $name<T> {
            type Output = $name<T>;
            fn $fn(self, other: $name<T>) -> $name<T> {
                $name {
                    $(
                        $fields: self.$fields $op other.$fields,
                    )*
                }
            }
        }
    }
}
