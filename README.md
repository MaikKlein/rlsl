# RLSL - Rust Like Shading Langauge (Highly experimental)

Experimental rust compiler from `mir` -> `spirv`.


Compile Rust in `compiler/rust` and create a new toolchain
``` 
rustup toolchain link rlsl_rust compiler/rust/build/x86_64-unknown-linux-gnu/stage2/
```
and override the current dir
``` 
rustup override set rlsl_rust
```


# Features


## `struct` and `enum`

Structs map 1:1 to SPIR-V

```
struct Foo {
    a: Type1,
    b: Type2,
    ...
}
```

Enum are also supported but are *not* implemented with untagged unions.


```
enum Test{
    Variant1(u32),
    Variant2(u32, u32)
}
```

Roughly translates to
```
struct Variant1(u32);
struct Variant2(u32, u32);
struct Test{
    _0: Variant1,
    _1: Variant2,
    dicr: IntType
}
```

It is advised to keep variants that contain data to a minimum.

## Function definition / calls
Functions translate 1:1 to SPIR-V
```
fn foo(a: u32) -> u32;
```

## Pointers

SPIR-V in logical addressing has no concept of pointers. It is not possible to store pointers inside structs.

``` 
struct Foo<'a>{
    // *Not* allowed
    bar: &'a Bar,
}
```

It is possible to keep a lot of pointer semantics.
```
let foo: &Foo = &foo;

fn foo_fn(foo: &Foo);

fn bar(&mut self);

fn get(&self) -> &Bar;
```

It is important to note that pointers are zero sized and are optimized behined to scenes. Essentially they
are just aliases for objects.

## Primitive (*Not a complete list*)

- `bool`
- `u32`
- `i32`
- `usize` (currently maps to u32)
- `isize` (currently maps to i32)
- `f32`
- `Vec2`

## Generics

Generics work as you would expect but dynamic dispatch is not supported.



