"""Generate a clamped C3D8 solid beam for FEMaster stiffness export."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nx", type=int, default=576)
    parser.add_argument("--ny", type=int, default=23)
    parser.add_argument("--nz", type=int, default=23)
    parser.add_argument("--length", type=float)
    parser.add_argument("--width", type=float)
    parser.add_argument("--height", type=float)
    args = parser.parse_args()
    nx, ny, nz = args.nx, args.ny, args.nz
    if min(nx, ny, nz) < 1:
        parser.error("element counts must be positive")
    lengths = (nx if args.length is None else args.length,
               ny if args.width is None else args.width,
               nz if args.height is None else args.height)
    if min(lengths) <= 0:
        parser.error("dimensions must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def node(i, j, k):
        return 1 + (i * (ny + 1) + j) * (nz + 1) + k

    with args.output.open("w") as out:
        out.write("** C3D8 cantilever, E=210000, nu=0.3\n*NODE\n")
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    out.write(f"{node(i, j, k)}, {i*lengths[0]/nx:.16g}, "
                              f"{j*lengths[1]/ny:.16g}, {k*lengths[2]/nz:.16g}\n")
        out.write("*ELEMENT, TYPE=C3D8, ELSET=SOLID\n")
        element = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    element += 1
                    nodes = [node(i, j, k), node(i+1, j, k),
                             node(i+1, j+1, k), node(i, j+1, k),
                             node(i, j, k+1), node(i+1, j, k+1),
                             node(i+1, j+1, k+1), node(i, j+1, k+1)]
                    out.write(", ".join(map(str, [element, *nodes])) + "\n")
        out.write("*MATERIAL, NAME=STEEL\n*ELASTIC, TYPE=ISOTROPIC\n210000, 0.3\n"
                  "*SOLID SECTION, ELSET=SOLID, MATERIAL=STEEL\n"
                  "*SUPPORT, SUPPORT_COLLECTOR=CLAMP\n")
        for j in range(ny + 1):
            for k in range(nz + 1):
                out.write(f"{node(0, j, k)}, 0, 0, 0\n")
        out.write("*CLOAD, LOAD_COLLECTOR=TIP\n")
        out.write(f"{node(nx, ny//2, nz//2)}, 0, 0, -1\n")
        out.write("*LOADCASE, TYPE=LINEARSTATIC\n*SUPPORTS\nCLAMP\n*LOADS\nTIP\n"
                  "*SOLVER, DEVICE=CPU, METHOD=DIRECT\n")
        # FEMaster normalizes keyword values to uppercase. A relative basename
        # works on case-sensitive filesystems when run in the output directory.
        out.write(f"*REQUESTSTIFFNESS, FILE={args.output.stem.upper()}\n*END\n")
    print(f"elements={element} total_dofs={3*(nx+1)*(ny+1)*(nz+1)} "
          f"free_dofs={3*nx*(ny+1)*(nz+1)} deck={args.output}")


if __name__ == "__main__":
    main()
