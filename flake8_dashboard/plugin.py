# -*- coding: utf-8 -*-
"""A plugin for flake8 to generate HTML dashboard reports."""
from collections import defaultdict

import codecs
import itertools
import json
import os
import pandas
import plotly
from bs4 import BeautifulSoup
from distutils import dir_util
from flake8.formatting import base
from jinja2 import Environment, PackageLoader
from jsmin import jsmin
from pathlib import PurePath

from flake8_dashboard.utils import (
    full_split,
    create_dir,
    ASTWalker,
    map_values_to_cmap,
    relative_path,
)

jinja2_env = Environment(loader=PackageLoader("flake8_dashboard"))

# A sequence of error code prefixes
# The first matching prefix determines the severity

SEVERITY_ORDER = [("E9", 1), ("F", 1), ("E", 2), ("W", 2), ("C", 2), ("D", 3)]
DEFAULT_SEVERITY = 3

SEVERITY_NAMES = dict()
SEVERITY_NAMES[1] = "High"
SEVERITY_NAMES[2] = "Medium"
SEVERITY_NAMES[3] = "Low"

# red, blue, green
SEVERITY_COLORS = dict(High="#EF553B", Medium="#636EFA", Low="#00CC96")

SEVERITY_DESCRIPTION = dict()
SEVERITY_DESCRIPTION[1] = "SyntaxError, IndentationError, IOError, and PyFlakes errors"
SEVERITY_DESCRIPTION[2] = "Pep8 and McCabe complexity errors"
SEVERITY_DESCRIPTION[3] = "Docstrings format errors"


def find_severity(code):
    """Given a flake8-style error code, return an ordinal severity."""
    for prefix, sev in SEVERITY_ORDER:
        if code.startswith(prefix):
            return sev
    return DEFAULT_SEVERITY


class DashboardReporter(base.BaseFormatter):
    """A plugin for flake8 to render errors as HTML reports."""

    def format(self, error):
        pass

    def __init__(self, *args, **kwargs):
        """ Initialize the formatter."""
        super().__init__(*args, **kwargs)

        self.html_output_dir = self.options.outputdir

        create_dir(self.html_output_dir)

        self.errors = []
        self.files_analized = 0

        self.code_description = defaultdict(lambda: "")
        code_descriptoion_file = os.path.join(
            os.path.dirname(__file__), "code_description.json"
        )
        with open(code_descriptoion_file) as json_file:
            self.code_description.update(json.load(json_file))

        self.statements_per_file = pandas.Series()
        self._astroid_walker = ASTWalker()

    def handle(self, error):
        """Record this error against the current file."""

        severity = find_severity(error.code)

        error_input = error._asdict()

        error_input["severity"] = severity

        self.errors.append(error_input)

    def beginning(self, filename):
        """Start processing a file"""
        self.files_analized += 1

    def finished(self, filename):
        """Finish the processing of a file."""

        self.statements_per_file[filename] = self._astroid_walker.count_statements(
            filename
        )

    def stop(self):
        """After the flake8 run, write the stylesheet and index."""

        # Save in a pandas dataframe all the errors
        params = {
            "total_errors": len(self.errors),
            "errors_found": len(self.errors) > 0,
            "files_analized": self.files_analized,
        }

        if params["total_errors"] > 0:
            error_db = pandas.DataFrame(
                self.errors,
                columns=[
                    "filename",
                    "code",
                    "line_number",
                    "text",
                    "column_number",
                    "physical_line",
                    "severity",
                ],
            )

            error_db.rename(columns={"filename": "path"}, inplace=True)

            file_list = error_db["path"].to_list()

            # Remove the project's root from the path
            common_prefix = os.path.commonprefix(file_list)
            if self.options.title is None:
                params["dashboard_title"] = os.path.abspath(common_prefix)
            else:
                params["dashboard_title"] = self.options.title
            error_db["path"] = error_db["path"].apply(
                lambda _path: relative_path(_path, common_prefix)
            )

            self.statements_per_file = self.statements_per_file.reset_index(
                name="statements"
            )
            self.statements_per_file.rename(columns={"index": "path"}, inplace=True)

            self.statements_per_file["path"] = self.statements_per_file["path"].apply(
                lambda _path: relative_path(_path, common_prefix)
            )

            ############################################################################
            # Pie plot with errors by severity

            error_severity = error_db.severity.apply(lambda x: SEVERITY_NAMES[x])
            error_severity = pandas.DataFrame(error_severity.value_counts())

            plot_id, plot_js = self._create_pie_plot_js(
                error_severity.index, error_severity.severity.values
            )
            params["id_severity_pie_plot"] = plot_id
            params["js_severity_pie_plot"] = plot_js

            errors_by_module = self._aggregate_by_folder_or_file(error_db)

            high_severity_by_folder = self._aggregate_by_folder_or_file(
                error_db[error_db["severity"] == 1]
            )
            medium_severity_warnings_by_folder = self._aggregate_by_folder_or_file(
                error_db[error_db["severity"] == 2]
            )
            low_severity_warnings_by_folder = self._aggregate_by_folder_or_file(
                error_db[error_db["severity"] == 3]
            )

            errors_by_module["errors"] = high_severity_by_folder["counts"]
            errors_by_module["warnings"] = medium_severity_warnings_by_folder["counts"]
            errors_by_module["low_severity"] = low_severity_warnings_by_folder["counts"]
            errors_by_module.fillna(0, inplace=True)
            del high_severity_by_folder, medium_severity_warnings_by_folder

            self.statements_per_file["n_files"] = 1

            ############################################################################
            # Sunburst plot of number of errors by directory and file
            statements_by_folder = self._aggregate_by_folder_or_file(
                self.statements_per_file, logic="sum"
            )
            errors_by_module.rename(columns={"counts": "error_counts"}, inplace=True)

            errors_by_module["statements"] = statements_by_folder["statements"]

            error_penalty = 5 * errors_by_module["errors"].astype(float)
            warnings_penalty = errors_by_module["warnings"].astype(float)
            low_severity_penalty = 0.25 * errors_by_module["low_severity"].astype(float)
            statements = errors_by_module["statements"].astype(float)

            penalty = (
                error_penalty + warnings_penalty + low_severity_penalty
            ) / statements

            errors_by_module["rating"] = 10.0 - 10 * penalty

            errors_by_module["rating"].fillna(0, inplace=True)
            errors_by_module["rating"] = errors_by_module["rating"].apply(
                lambda x: max(0, x)
            )

            plot_id, plot_js = self._create_sunburst_plot_js(
                parents=errors_by_module["parent"],
                values=errors_by_module["error_counts"],
                ids=errors_by_module["path"],
                labels=errors_by_module["label"],
                textinfo="label + value + text",
            )

            params["id_plot_error_by_folder"] = plot_id
            params["js_plot_error_by_folder"] = plot_js

            ############################################################################
            # Sunburst plot of number of errors by code
            errors_by_code = self._aggregate_by_code(error_db, self.code_description)
            errors_by_code.rename(columns={"counts": "error_counts"}, inplace=True)

            plot_id, plot_js = self._create_sunburst_plot_js(
                parents=errors_by_code["parent"],
                values=errors_by_code["error_counts"],
                ids=errors_by_code["code"],
                labels=errors_by_code["code"],
                text=errors_by_code["code_description"].values,
                textinfo="label + value + text",
            )
            params["id_plot_error_by_code"] = plot_id
            params["js_plot_error_by_code"] = plot_js

            ############################################################################
            # Sunburst plot with quality score for each directory, module, and file

            errors_by_module["level"] = errors_by_module["path"].apply(
                lambda _path: len(PurePath(_path).parts)
            )

            errors_by_module.loc["All", "level"] = 0
            errors_by_module.sort_values(["level"], inplace=True)

            # Before plotting we compute each sector size in a way that at each level,
            # the sector size is the same for all the elements in that level.
            n_childs = errors_by_module["parent"].value_counts()

            sector_by_parent = n_childs.copy()
            for path, parent in errors_by_module["parent"].iteritems():
                if parent == "":
                    sector_by_parent[parent] = 1
                    errors_by_module.loc[path, "sector_size"] = 1000
                elif parent == "All":
                    errors_by_module.loc[path, "sector_size"] = 1000 / n_childs[parent]

                else:
                    errors_by_module.loc[path, "sector_size"] = (
                        errors_by_module.loc[parent, "sector_size"] / n_childs[parent]
                    )
            # The resulting sector sizes have rounding errors due to the floating point
            # operations.
            # But, the sunburst plot needs the total value for node (parent) bigger or
            # equal than the sum of its children.
            # In some cases, the values were slightly smaller and the plot was not
            # shown.
            # This is a fix that by adding a small margin to each node.
            errors_by_module.sort_values(["level"], ascending=False, inplace=True)
            levels = errors_by_module["level"]
            max_level = levels.max()

            for parent in errors_by_module.parent.unique():

                if parent == "":
                    continue

                percentual_diff = (max_level - levels[parent]) * 1e-5
                diff_sector_size = (
                    errors_by_module.loc[parent, "sector_size"] * percentual_diff
                )

                errors_by_module.loc[parent, "sector_size"] += diff_sector_size

            plot_id, plot_js = self._create_sunburst_plot_js(
                parents=errors_by_module["parent"],
                values=errors_by_module["sector_size"],
                ids=errors_by_module["path"],
                labels=errors_by_module["label"],
                text=errors_by_module["rating"].map(lambda x: f"{x:.1f}"),
                maxdepth=2,
                hoverinfo="label+text",
                marker={
                    "line": {"width": 2},
                    "colors": map_values_to_cmap(
                        errors_by_module["rating"].values / 10
                    ),
                },
                layout_kwargs=dict(
                    xaxis=dict(visible=False, showgrid=False),
                    yaxis=dict(visible=False, showgrid=False, showline=False),
                    plot_bgcolor="#fff",
                    font=dict(size=18, color="#7f7f7f"),
                    legend={
                        "orientation": "h",
                        "font": dict(size=22),
                        "x": 0.5,
                        "y": 1.02,
                        "xanchor": "center",
                        "yanchor": "bottom",
                        "itemclick": False,
                    },
                ),
            )

            params["id_plot_code_rating"] = plot_id
            params["js_plot_code_rating"] = plot_js

            self.write_index(params)

            if self.options.debug:
                file_path = os.path.join(self.options.outputdir, "quality.csv")
                errors_by_module.to_csv(file_path)
                print("Debug file saved: " + file_path)

                file_path = os.path.join(self.options.outputdir, "report.csv")
                error_db.to_csv(file_path)
                print("Debug file saved: " + file_path)

    def write_index(self, params):
        report_template = jinja2_env.get_template("index.html")
        rendered = report_template.render(params)

        report_filename = os.path.join(self.html_output_dir, "index.html")
        with codecs.open(report_filename, "w", encoding="utf8") as f:
            f.write(rendered)

        dir_util.copy_tree(
            os.path.join(os.path.dirname(__file__), "templates", "assets"),
            os.path.join(self.html_output_dir, "assets"),
        )

    @staticmethod
    def _create_pie_plot_js(labels, values):
        colors = [SEVERITY_COLORS[sev] for sev in labels]
        trace = plotly.graph_objs.Pie(
            labels=labels,
            values=values,
            hoverinfo="label+text",
            textinfo="text",
            text=[
                f"{value}<br>({value * 100 / values.sum():.3g}%)" for value in values
            ],
            textfont=dict(size=22),
            marker={"line": {"width": 2}, "colors": colors},
        )

        layout = plotly.graph_objs.Layout(
            margin=plotly.graph_objs.layout.Margin(t=1, l=1, r=1, b=1), showlegend=False
        )

        fig = plotly.graph_objs.Figure([trace], layout)

        div = plotly.offline.plot(
            fig,
            config={
                "responsive": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["sendDataToCloud", "hoverClosestPie"],
            },
            include_plotlyjs=False,
            output_type="div",
        )

        soup = BeautifulSoup(div, features="html.parser")
        return soup.div.div["id"], jsmin(soup.div.script.decode_contents())

    @staticmethod
    def _create_sunburst_plot_js(
        parents=None,
        values=None,
        ids=None,
        labels=None,
        text=None,
        maxdepth=3,
        **kwargs,
    ):

        extra_traces = kwargs.pop("extra_traces", list())
        layout_kwargs = kwargs.pop("layout_kwargs", dict())
        trace = plotly.graph_objs.Sunburst(
            parents=parents,
            values=values,
            ids=ids,
            labels=labels,
            text=text,
            branchvalues="total",
            maxdepth=maxdepth,
            **kwargs,
        )

        layout = plotly.graph_objs.Layout(
            margin=plotly.graph_objs.layout.Margin(t=0, l=0, r=0, b=0), **layout_kwargs
        )
        fig = plotly.graph_objs.Figure([trace] + extra_traces, layout)

        div = plotly.offline.plot(
            fig,
            config={
                "responsive": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["sendDataToCloud", "toggleHover"],
            },
            include_plotlyjs=False,
            output_type="div",
        )

        soup = BeautifulSoup(div, features="html.parser")
        return soup.div.div["id"], jsmin(soup.div.script.decode_contents())

    @staticmethod
    def _aggregate_by_folder_or_file(error_db, logic="counts"):
        """
        Aggregate DataFrame by module/directory and by file.
        The input DataFrame must contain only file information.
        """

        if error_db.shape[0] == 0:
            return pandas.DataFrame(
                columns=["path", "counts", "parent", "id"]
            ).set_index("path", drop=False)

        files_names = error_db["path"].drop_duplicates()

        if logic == "counts":
            total_errors_by_file = error_db["path"].value_counts()
            total_errors_by_file = pandas.DataFrame(
                data=total_errors_by_file.values,
                index=total_errors_by_file.index,
                columns=["counts"],
            )
        else:
            _error_db = error_db.copy()
            _error_db["counts"] = 1
            total_errors_by_file = _error_db.groupby("path").sum()

        total_errors_by_file["n_files"] = 1
        aggregated_by_folder = total_errors_by_file.copy()

        intermediate_paths = files_names.apply(full_split)

        intermediate_paths = set(
            itertools.chain.from_iterable(intermediate_paths.values)
        )

        for _path in intermediate_paths:
            sel = total_errors_by_file.index.get_level_values(0).str.match(f"^{_path}/")
            aggregated_by_folder.loc[_path] = total_errors_by_file[sel].sum()

        aggregated_by_folder["path"] = aggregated_by_folder.index

        aggregated_by_folder.reset_index(inplace=True, drop=True)

        tmp = aggregated_by_folder["path"].apply(os.path.split)
        aggregated_by_folder["parent"] = tmp.apply(lambda x: x[0])
        # Keep the last directory/module as id
        aggregated_by_folder["label"] = tmp.apply(lambda x: x[1])

        sel = aggregated_by_folder["parent"] == ""  # Select root path

        aggregated_by_folder.loc[sel, "parent"] = "All"
        aggregated_by_folder.loc[sel, "label"] = aggregated_by_folder.loc[sel, "path"]

        aggregated_by_folder.set_index("path", drop=False, inplace=True)
        sel = aggregated_by_folder["parent"] == "All"
        aggregated_by_folder.loc["All"] = aggregated_by_folder.loc[sel].sum()
        aggregated_by_folder.loc["All", "path"] = "All"
        aggregated_by_folder.loc["All", "label"] = "All"
        aggregated_by_folder.loc["All", "parent"] = ""

        return aggregated_by_folder

    @staticmethod
    def _aggregate_by_code(error_db, code_description):
        """Compute the total number of errors aggregated by error code."""

        error_codes = error_db.code.value_counts()
        error_codes = error_codes.reset_index(name="counts")
        error_codes.rename(columns={"index": "code"}, inplace=True)
        parents = error_codes["code"].apply(lambda x: (x[0], x[:2]))
        parents = set(itertools.chain.from_iterable(parents.values))

        aggregated_by_code = pandas.DataFrame(columns=error_codes.columns)
        for i, parent in enumerate(parents):
            sel = error_codes.code.str.match(f"^{parent}")
            row = [parent, error_codes[sel]["counts"].sum()]
            aggregated_by_code.loc[i + 1, ["code", "counts"]] = row

        row = ["", error_codes["counts"].sum()]
        aggregated_by_code.loc[0, ["code", "counts"]] = row

        aggregated_by_code = pandas.concat(
            [error_codes, aggregated_by_code], sort=True
        ).reset_index(drop=True)

        def get_parent(code):
            if len(code) > 2:
                return code[:2]
            elif len(code) == 2:
                return code[0]
            elif len(code) == 1:
                return "All"
            else:
                return ""

        parents = aggregated_by_code["code"].apply(get_parent)
        aggregated_by_code["parent"] = parents

        aggregated_by_code.loc[aggregated_by_code["code"] == "", "code"] = "All"
        aggregated_by_code.sort_values(["parent"], inplace=True)

        aggregated_by_code["code_description"] = aggregated_by_code["code"].apply(
            lambda x: code_description[x]
        )
        aggregated_by_code["severity"] = (
            aggregated_by_code["code"]
            .apply(find_severity)
            .apply(lambda x: SEVERITY_NAMES[x])
        )

        return aggregated_by_code

    @classmethod
    def add_options(cls, options):
        """Add options to the OptionsManager."""

        cls.option_manager = options

        options.add_option(
            "--outputdir",
            help="Directory to save the HTML output.",
            parse_from_config=True,
            default="./flake8_dashboard",
        )

        options.add_option(
            "--debug",
            help="Write additional debugging information as csv format.",
            parse_from_config=True,
            default=False,
            action="store_true",
        )

        options.add_option(
            "--title",
            help="Set the dashboard's title. No title by default.",
            parse_from_config=True,
            default=None,
        )
