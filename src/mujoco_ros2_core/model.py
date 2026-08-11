"""Prepare vendor URDF files for MuJoCo without modifying their source."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class PreparedModel:
    source: Path
    output: Path
    sha256: str
    replacements: int


def prepare_urdf(
    source: str | Path,
    output: str | Path,
    *,
    package_map: dict[str, str | Path],
    preserve_visuals: bool = True,
    fuse_static: bool = False,
) -> PreparedModel:
    """Resolve ROS package URIs and add deterministic MuJoCo compiler options."""

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source URDF does not exist: {source_path}")

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(source_path, parser=parser)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"expected URDF <robot> root, got <{root.tag}>")

    resolved_packages = {
        name: Path(path).expanduser().resolve() for name, path in package_map.items()
    }
    replacements = 0
    for element in root.iter():
        for attribute, value in tuple(element.attrib.items()):
            if not value.startswith("package://"):
                continue
            package_and_path = value[len("package://") :]
            package, separator, relative = package_and_path.partition("/")
            if not separator or package not in resolved_packages:
                raise ValueError(f"no package mapping for URI: {value}")
            element.set(attribute, str(resolved_packages[package] / relative))
            replacements += 1

    mujoco_element = root.find("mujoco")
    if mujoco_element is None:
        mujoco_element = ET.SubElement(root, "mujoco")
    compiler = mujoco_element.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_element, "compiler")
    compiler.set("discardvisual", "false" if preserve_visuals else "true")
    compiler.set("fusestatic", "true" if fuse_static else "false")

    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return PreparedModel(source_path, output_path, digest, replacements)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Map a ROS package name to its asset directory; repeat as needed.",
    )
    parser.add_argument("--discard-visuals", action="store_true")
    parser.add_argument("--fuse-static", action="store_true")
    args = parser.parse_args()

    package_map: dict[str, Path] = {}
    for item in args.package:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            parser.error(f"invalid --package value: {item!r}; expected NAME=PATH")
        package_map[name] = Path(path)

    result = prepare_urdf(
        args.source,
        args.output,
        package_map=package_map,
        preserve_visuals=not args.discard_visuals,
        fuse_static=args.fuse_static,
    )
    print(
        f"prepared={result.output} replacements={result.replacements} "
        f"sha256={result.sha256}"
    )


if __name__ == "__main__":
    main()

