import sys
from pathlib import Path


def parse_mps(path):

    section = None

    sense = "minimize"

    rows = []
    row_type = {}
    row_rhs = {}

    columns = []

    variables = []
    variable_set = set()

    bounds = {}

    objective_name = None

    lines = Path(path).read_text(
        errors="replace"
    ).splitlines()

    for raw in lines:

        line = raw.strip()

        if not line:
            continue

        if line.startswith("*"):
            continue

        upper = line.upper()

        if upper == "NAME" or upper.startswith("NAME "):
            section = "NAME"
            continue

        if upper == "OBJSENSE":
            section = "OBJSENSE"
            continue

        if upper == "MIN":
            sense = "minimize"
            continue

        if upper == "MAX":
            sense = "maximize"
            continue

        if upper == "ROWS":
            section = "ROWS"
            continue

        if upper == "COLUMNS":
            section = "COLUMNS"
            continue

        if upper == "RHS":
            section = "RHS"
            continue

        if upper == "BOUNDS":
            section = "BOUNDS"
            continue

        if upper == "RANGES":
            section = "RANGES"
            continue

        if upper == "ENDATA":
            break

        parts = line.split()

        if section == "ROWS":

            if len(parts) >= 2:

                row_kind = parts[0].upper()
                row_name = parts[1]

                if row_kind == "N":

                    objective_name = row_name

                else:

                    row_type[row_name] = row_kind
                    rows.append(row_name)


        elif section == "COLUMNS":

            if len(parts) >= 3:

                variable = parts[0]

                if variable not in variable_set:

                    variable_set.add(variable)
                    variables.append(variable)

                pairs = parts[1:]

                for i in range(
                    0,
                    len(pairs) - 1,
                    2
                ):

                    row_name = pairs[i]

                    value = float(
                        pairs[i + 1]
                    )

                    columns.append(
                        (
                            variable,
                            row_name,
                            value
                        )
                    )


        elif section == "RHS":

            pairs = parts[1:]

            for i in range(
                0,
                len(pairs) - 1,
                2
            ):

                row_name = pairs[i]

                value = float(
                    pairs[i + 1]
                )

                row_rhs[row_name] = value


        elif section == "BOUNDS":

            if len(parts) >= 3:

                bound_type = parts[0].upper()

                variable = parts[2]

                bounds.setdefault(
                    variable,
                    {
                        "lo": 0.0,
                        "hi": None
                    }
                )

                if bound_type in (
                    "LO",
                    "LI"
                ):

                    bounds[variable]["lo"] = (
                        float(parts[3])
                        if len(parts) > 3
                        else 0.0
                    )

                elif bound_type in (
                    "UP",
                    "UI"
                ):

                    bounds[variable]["hi"] = float(
                        parts[3]
                    )

                elif bound_type == "FX":

                    value = float(parts[3])

                    bounds[variable] = {
                        "lo": value,
                        "hi": value
                    }

                elif bound_type == "FR":

                    bounds[variable] = {
                        "lo": None,
                        "hi": None
                    }

                elif bound_type == "MI":

                    bounds[variable]["lo"] = None

                elif bound_type == "PL":

                    bounds[variable]["hi"] = None

                elif bound_type == "BV":

                    bounds[variable] = {
                        "lo": 0.0,
                        "hi": 1.0
                    }


    row_index = {
        row: i
        for i, row in enumerate(rows)
    }

    variable_index = {
        variable: i
        for i, variable in enumerate(variables)
    }


    matrix = [
        [0.0 for _ in variables]
        for _ in rows
    ]

    objective = [
        0.0 for _ in variables
    ]


    for variable, row, value in columns:

        j = variable_index[variable]

        if row == objective_name:

            objective[j] += value

        elif row in row_index:

            i = row_index[row]

            matrix[i][j] += value


    senses = []

    for row in rows:

        if row_type[row] == "L":

            senses.append("<=")

        elif row_type[row] == "G":

            senses.append(">=")

        else:

            senses.append("=")


    lower_bounds = []

    upper_bounds = []

    for variable in variables:

        b = bounds.get(
            variable,
            {
                "lo": 0.0,
                "hi": None
            }
        )

        lower_bounds.append(
            b["lo"]
        )

        upper_bounds.append(
            b["hi"]
        )


    return {
        "sense": sense,
        "variables": variables,
        "objective": objective,
        "constraints": [
            {
                "name": rows[i],
                "coefficients": matrix[i],
                "sense": senses[i],
                "rhs": row_rhs.get(
                    rows[i],
                    0.0
                )
            }
            for i in range(len(rows))
        ],
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds
    }


def main():

    if len(sys.argv) != 3:

        print(
            "Usage: python mps_converter.py "
            "input.mps output.lpdata",
            file=sys.stderr
        )

        return 1


    input_file = sys.argv[1]

    output_file = sys.argv[2]


    data = parse_mps(
        input_file
    )


    variables = data["variables"]

    constraints = data["constraints"]


    lines = []

    lines.append(
        "VARIABLES "
        + str(len(variables))
    )

    lines.append(
        "CONSTRAINTS "
        + str(len(constraints))
    )

    lines.append(
        "VARIABLE_NAMES "
        + " ".join(variables)
    )

    lines.append(
        "OBJECTIVE "
        + " ".join(
            map(
                str,
                data["objective"]
            )
        )
    )

    lines.append(
        "MATRIX"
    )


    for constraint in constraints:

        lines.append(
            "ROW "
            + constraint["name"]
            + " "
            + constraint["sense"]
            + " "
            + str(
                constraint["rhs"]
            )
            + " "
            + " ".join(
                map(
                    str,
                    constraint["coefficients"]
                )
            )
        )


    lines.append(
        "END"
    )


    Path(output_file).write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


    print(
        f"Converted "
        f"{len(variables)} variables, "
        f"{len(constraints)} constraints"
    )


    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )