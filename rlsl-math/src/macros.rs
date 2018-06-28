macro_rules! vec_op_vec {
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

macro_rules! vec_ops_vec {
    ($name: ident {$($fields:ident),*}) => {
        vec_op_vec!($name {$($fields),*}, Add, add, +);
        vec_op_vec!($name {$($fields),*}, Sub, sub, -);
        vec_op_vec!($name {$($fields),*}, Div, div, /);
        vec_op_vec!($name {$($fields),*}, Mul, mul, *);
    }
}

macro_rules! vec_op_scalar {
    ($name: ident {$($fields:ident),*}, $trait: ident, $fn: ident, $op: tt) => {
        impl<T: Float> ::std::ops::$trait<T> for $name<T> {
            type Output = $name<T>;
            fn $fn(self, scalar: T) -> $name<T> {
                $name {
                    $(
                        $fields: self.$fields $op scalar,
                    )*
                }
            }
        }
    }
}

macro_rules! vec_ops_scalar {
    ($name: ident {$($fields:ident),*}) => {
        vec_op_scalar!($name {$($fields),*}, Add, add, +);
        vec_op_scalar!($name {$($fields),*}, Sub, sub, -);
        vec_op_scalar!($name {$($fields),*}, Div, div, /);
        vec_op_scalar!($name {$($fields),*}, Mul, mul, *);
    }
}
