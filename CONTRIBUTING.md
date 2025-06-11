# Contributing Guidelines

The following items provide guidance for developers who wish to contribute to `jQMC`.

## Principles

In `jQMC`, our top priorities are:

1. **Sustainability** – maintainability of the codebase
2. **Ease of development** – simplicity of implementing new features or theories

We are willing to sacrifice some computational speed to achieve these goals. To that end, please follow these guidelines when contributing:

---

### Data and Algorithms

* `jQMC` is written in `Python` but uses a style that differs from conventional object-oriented programming (OOP).
* **Separation of concerns**:

  * **Data** are defined as static classes using `flax.struct.dataclass`.
  * **Algorithms** are implemented as standalone functions that accept these dataclass instances as arguments.
* Related dataclasses and their algorithms live in the same `Python` file and module, preserving the spirit of OOP while ensuring clarity.
* This design not only improves readability but also aligns with `JAX`’s requirement for side-effect-free functions.

---

### Testing

* Robust testing is central to `jQMC`. For every functionality, `jQMC` provides two implementations:

  1. A **`_debug`** version, written for human readability and easy tracing of the logic flow.
  2. A **`_jax`** version, optimized and decorated with `@jit` for high performance (though its control flow may be less obvious).
* In **`pytest`**, we verify that `_debug` and `_jax` produce numerically identical results.
* **When adding any new method**, you must implement both `_debug` and `_jax` variants. Pull requests lacking either will not be approved.

---

### GitHub Actions

* We use GitHub Actions to run `pytest` automatically on `merge` and `pull_request` events.
* If you add new tests, remember to update the corresponding workflow scripts so they are executed in CI.

---

### Examples and Tutorials

* Whenever you introduce a new computational feature, create a tutorial or example under the `examples/` directory to demonstrate its usage alongside your tests.

---

### Merging into the Main Branch

* Submit a **Pull Request** (PR).
* Upon PR creation or update, GitHub Actions will run the test suite.
* If all tests pass, @kousuke-nakano (a main maintainer) will review your changes.
* Once approved, your PR will be merged into `main`.

---

### Dependent Modules

* To balance future maintainability with development efficiency, we strive to minimize external dependencies.
* **Core dependencies** (outside the Python standard library) are currently limited to:

  * `numpy`
  * `scipy`
  * `jax`
  * `flax`
* Other third-party packages should be avoided unless absolutely necessary. Any new dependency must be approved by @kousuke-nakano.

---

### Release Process

* All official package releases are performed by @kousuke-nakano as needed.

---

Thank you for considering contributing your work to the `jQMC` code!
