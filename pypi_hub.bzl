def _normalize_name(name):
    return name.lower().replace("-", "_").replace(".", "_")

def _pypi_hub_repo_impl(rctx):
    lock_content = rctx.read(rctx.attr.requirements_lock)
    packages = {}
    for line in lock_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" in line:
            pkg_raw = line.split("==")[0].strip()
            if "[" in pkg_raw:
                pkg_raw = pkg_raw.split("[")[0].strip()
            norm = _normalize_name(pkg_raw)
            packages[norm] = True

    # Root BUILD.bazel and REPO.bazel
    rctx.file("BUILD.bazel", "package(default_visibility = [\"//visibility:public\"])\n")
    rctx.file("REPO.bazel", "")

    for pkg in packages.keys():
        if pkg in ["torch", "torchvision"]:
            content = """package(default_visibility = ["//visibility:public"])

alias(
    name = "{pkg}",
    actual = ":pkg",
)

alias(
    name = "pkg",
    actual = select({{
        "@//:is_rocm_backend": "@pypi_rocm//{pkg}:pkg",
        "//conditions:default": "@pypi_cuda//{pkg}:pkg",
    }}),
)
""".format(pkg = pkg)
        else:
            content = """package(default_visibility = ["//visibility:public"])

alias(
    name = "{pkg}",
    actual = ":pkg",
)

alias(
    name = "pkg",
    actual = "@pypi_cuda//{pkg}:pkg",
)
""".format(pkg = pkg)

        rctx.file("{}/BUILD.bazel".format(pkg), content)

pypi_hub_repo = repository_rule(
    implementation = _pypi_hub_repo_impl,
    attrs = {
        "requirements_lock": attr.label(mandatory = True),
    },
)

def _pypi_selector_impl(module_ctx):
    pypi_hub_repo(
        name = "pypi",
        requirements_lock = "//:requirements_lock_cuda.txt",
    )

pypi_selector = module_extension(
    implementation = _pypi_selector_impl,
)
