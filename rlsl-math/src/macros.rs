#[macro_export]
macro_rules! variadic2{
    ($f:path, $e1:expr) => {
        $e1
    };
    ($f:path, $e1:expr, $e2:expr) => {
        $f($e1, $e2)
    };
    ($f:path, $e1:expr, $e2:expr, $($rest:expr),*) => {
        $f(variadic2!($f, $e1, $e2), variadic2!($f, $($rest),*))
    };
}

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

macro_rules! vec_common {
    ($name: ident {$($fields:ident),*}) => {
        impl<T: Float> $name<T> {
            #[inline]
            pub fn new($( $fields: T, )*) -> $name<T> {
                $name {
                    $(
                        $fields,
                    )*
                }
            }
            #[inline]
            pub fn lerp(self, other: Self, t: T) -> Self {
                let i_t = T::one() - t;
                $(
                    let $fields = i_t * self.$fields + t * other.$fields;
                )*
                $name {
                    $(
                        $fields,
                    )*
                }
            }
            #[inline]
            pub fn single(t: T) -> Self {
                $name {
                    $(
                        $fields: t,
                    )*
                }
            }

            #[inline]
            pub fn map<R, F>(self, mut f: F) -> $name<R>
                where
                F: FnMut(T) -> R {
                    $name {
                        $(
                            $fields: f(self.$fields),
                        )*
                    }
            }

            pub fn any<F: FnMut(T) -> bool>(self, mut f: F) -> bool {
                use std::ops::BitOr;
                variadic2!(bool::bitor, $(f(self.$fields)),*) 
            }

            pub fn all<F: FnMut(T) -> bool>(self, mut f: F) -> bool {
                use std::ops::BitAnd;
                variadic2!(bool::bitand, $(f(self.$fields)),*) 
            }

            pub fn fold<R, F: FnMut(R, T) -> R>(self, acc: R, mut f: F) -> R {
                $(
                    let acc = f(acc, self.$fields);
                )*
                acc
            }

            pub fn add(self, other: Self) -> Self {
                self + other
            }

            pub fn sub(self, other: Self) -> Self {
                self - other
            }

            pub fn mul(self, other: Self) -> Self {
                self * other
            }
            pub fn div(self, other: Self) -> Self {
                self / other
            }

            pub fn dot(self, other: Self) -> T {
                <Self as Vector>::dot(self, other)
            }

            pub fn length(self) -> T {
                <Self as Vector>::length(self)

            }

            pub fn normalize(self) -> Self {
                self / self.length()
            }
        }


    }
}
