# Rlsl - Rust Like Shading Language

**The project is currently unusable**

[![Join the chat at https://gitter.im/MaikKlein/rlsl](https://badges.gitter.im/MaikKlein/rlsl.svg)](https://gitter.im/MaikKlein/rlsl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) 
[![Backers on Open Collective](https://opencollective.com/rlsl/backers/badge.svg)](#backers) 
[![Sponsors on Open Collective](https://opencollective.com/rlsl/sponsors/badge.svg)](#sponsors) 

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

## Contributors

This project exists thanks to all the people who contribute.
<a href="graphs/contributors"><img src="https://opencollective.com/rlsl/contributors.svg?width=890&button=false" /></a>


## Backers

Thank you to all our backers! 🙏 [[Become a backer](https://opencollective.com/rlsl#backer)]

<a href="https://opencollective.com/rlsl#backers" target="_blank"><img src="https://opencollective.com/rlsl/backers.svg?width=890"></a>


## Sponsors

Support this project by becoming a sponsor. Your logo will show up here with a link to your website. [[Become a sponsor](https://opencollective.com/rlsl#sponsor)]

<a href="https://opencollective.com/rlsl/sponsor/0/website" target="_blank"><img src="https://opencollective.com/rlsl/sponsor/0/avatar.svg"></a>
<a href="https://opencollective.com/rlsl/sponsor/1/website" target="_blank"><img src="https://opencollective.com/rlsl/sponsor/1/avatar.svg"></a>
<a href="https://opencollective.com/rlsl/sponsor/2/website" target="_blank"><img src="https://opencollective.com/rlsl/sponsor/2/avatar.svg"></a>
<a href="https://opencollective.com/rlsl/sponsor/3/website" target="_blank"><img src="https://opencollective.com/rlsl/sponsor/3/avatar.svg"></a>
<a href="https://opencollective.com/rlsl/sponsor/4/website" target="_blank"><img src="https://opencollective.com/rlsl/sponsor/4/avatar.svg"></a>
<a href="https://opencollective.com/rlsl/sponsor/5/website" target="_blank"><img src="https://opencollective.com/rlsl/sponsor/5/avatar.svg"></a>
<a href="https://opencollective.com/rlsl/sponsor/6/website" target="_blank"><img src="https://opencollective.com/rlsl/sponsor/6/avatar.svg"></a>
<a href="https://opencollective.com/rlsl/sponsor/7/website" target="_blank"><img src="https://opencollective.com/rlsl/sponsor/7/avatar.svg"></a>
<a href="https://opencollective.com/rlsl/sponsor/8/website" target="_blank"><img src="https://opencollective.com/rlsl/sponsor/8/avatar.svg"></a>
<a href="https://opencollective.com/rlsl/sponsor/9/website" target="_blank"><img src="https://opencollective.com/rlsl/sponsor/9/avatar.svg"></a>


