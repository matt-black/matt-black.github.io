---
title: "winx"
location: "python, jax"
description: "N-dimensional windowed filtering for JAX. Application of array filters that compute over the entire window (e.g. 3x3 median-filtering an image) cannot be done in a jit-able, jax-transformable way. The closest mechanism is jax.lax.reduce_window but this requires the function to be a binary operation that compares elements within the window. winx provides this missing functionality, allowing users to construct and apply arbitrary, window-based filters to an n-dimensional array."
startDate: "2023-11-23"
sourceLink: "https://github.com/matt-black/winx"
---
