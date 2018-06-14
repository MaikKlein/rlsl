<p align="center">
  <img src="media/logo.png">
</p>

# Rlsl - Rust Like Shading Language

**The project is currently unusable**

[![Join the chat at https://gitter.im/MaikKlein/rlsl](https://badges.gitter.im/MaikKlein/rlsl.svg)](https://gitter.im/MaikKlein/rlsl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## What is Rlsl?
Rlsl can compile a subset of Rust to [SPIR-V](https://www.khronos.org/registry/spir-v/). You can read more about the limitations [here](https://github.com/MaikKlein/rlsl/wiki/Implementation-details).

Rlsl targets the [logical addressing model](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_memorymodelsection_a_memory_model) of SPIR-V.
>The Logical addressing model means pointers are abstract, having no physical size or numeric value. In this mode, pointers can only be created from existing objects, and they cannot be stored into an object, unless additional capabilities, e.g., VariablePointers, are declared to add such functionality.

## Features

* Supports cargo
* Multiple entry points can be defined in the same SPIR-V module
* Currently supports Vertex, Fragment and Compute shaders
* Shader code can run on the CPU because rlsl is a subset of Rust
* Reflection *TODO*
* Support library for interop between Rust and rlsl for uniforms (std140, std420) *TODO*

## Installation
 *TODO*

## How?

```
RUSTC=rlsl cargo build
```

![compile](https://raw.githubusercontent.com/MaikKlein/rlsl/master/media/compile.gif)

## Blog

1. [What is RLSL](https://maikklein.github.io/rlsl-progress-report/)
2. [Milestone 1](https://maikklein.github.io/rlsl-milestone-1/)

## Want to help?

### Contribute

The project currently does not accept any contributions yet.

* Rlsl can not be easily built by anyone
* There is no documentation
* Debugging tools are almost non existent
* There is no infrastructure for testing
* No guide level explanation for contributions

Rlsl will start to accept contributions after those issues are properly addressed.

### Donate

[![Patreon](https://c5.patreon.com/external/logo/become_a_patron_button.png)](https://www.patreon.com/maikklein)

## Community

Want to chat? Join us on [gitter](https://gitter.im/MaikKlein/rlsl).

Feel free to open an [issue](https://github.com/MaikKlein/rlsl/issues) at any time.
