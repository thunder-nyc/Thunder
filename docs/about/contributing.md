# Constributing to Thunder

Thanks a lot for your intersts in contributing to Thunder!

Please take a moment to read through this document in order to make the contribution process efficient for our developer community.

## C++ Style Guide

All of the source code contributed to Thunder **must follow strictly the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)**, with the following exceptions

1. Using exception is allowed, but only limited to those types in the `thunder::exception` namespace. If you think that none of the classes in `thunder::exception` covers your case, request a feature in our issue tracker.
2. All function names are in lower camel case, such as `SomeClass::doSomething()`, instead of the upper camel case `SomeClass::DoSomething()` in the Google style guide. This is because Thunder is a mathematical library -- writing `Tanh()` instead of `tanh()` feels more or less weird.

Feel free to contact the developers if you need clarification for the coding style.

## Issue Tracker

The development of Thunder is centered around our [issue tracker](https://github.com/thunder-nyc/Thunder/issues) provided by Github. If you have a feature request or bug report, you are welcome to report it there. Here are the guidelines for the issue tracker

1. If you have a feature request or bug report, search the issue tracker first to make sure it is not a duplication.

2. Check whether the issue has been fixed in the development branch of the repository.

3. If your requested feature or reported bug is new, feel free to create an issue.

For bug reports, it would be great if a minimal test case can be provided. Keep in mind that reproducing your issue is necessary for our developer community to fix a problem.

## Pull Requests

We welcome your pull requests that fix bugs, implement new feature or offer improvements to Thunder!
