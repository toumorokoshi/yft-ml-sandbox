# development

## setting up vscode

1. use the `create environment` command in python extension to create an environmen with all dependencies installed.
   1. use the `requirements.in` file to match bazel's dependencies.

## using ROCM

To use ROCM:

1. Make sure rocm is installed on your system
2. You will have to add the current user to the `render` or `video` group, and restart the OS to pick up those permission change.
3. Uncomment the appropriate lines in requirements.txt
4. run `bazel run "//requirements.update"`. This will take a really long time due to downloading the 4GB torch package compiled for rocm.