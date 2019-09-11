# -*- coding: utf-8 -*-
"""A plugin for flake8 to generate HTML reports."""
import codecs
import os
from collections import namedtuple

import itertools
import pandas
import plotly
from bs4 import BeautifulSoup
from flake8.formatting import base
from jinja2 import Environment, PackageLoader
from jsmin import jsmin

jinja2_env = Environment(
    loader=PackageLoader('flake8_reporters')
)

# A sequence of error code prefixes
# The first matching prefix determines the severity

SEVERITY_ORDER = [
    ('E9', 1),
    ('F', 1),
    ('E', 2),
    ('W', 2),
    ('C', 2),
    ('D', 3)
]
DEFAULT_SEVERITY = 3

SEVERITY_NAMES = [
    'high',
    'medium',
    'low'
]


def find_severity(code):
    """Given a flake8-style error code, return an ordinal severity."""
    for prefix, sev in SEVERITY_ORDER:
        if code.startswith(prefix):
            return sev
    return DEFAULT_SEVERITY


IndexEntry = namedtuple(
    'IndexEntry',
    'filename report_name error_count highest_sev'
)


def full_split(_path):
    intermediate_paths = list()
    tail = _path

    while len(tail) > 0:
        head, tail = os.path.split(_path)
        _path = head
        if len(_path) > 0:
            intermediate_paths.append(_path)

    return intermediate_paths


class Reporter(base.BaseFormatter):
    """A plugin for flake8 to render errors as HTML reports."""

    def after_init(self):
        """Configure the plugin run."""
        self.outdir = self.options.htmldir
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        self.errors = []

    def handle(self, error):
        """Record this error against the current file."""

        severity = find_severity(error.code)

        error_input = error._asdict()

        error_input['severity'] = severity

        self.errors.append(error_input)

    def finished(self, filename):
        """Write the HTML reports for filename."""
        pass

    def stop(self):
        """After the flake8 run, write the stylesheet and index."""

        # Save in a pandas dataframe all the errors
        error_db = pandas.DataFrame(self.errors,
                                    columns=['filename',
                                             'code',
                                             'line_number',
                                             'text',
                                             'column_number',
                                             'physical_line',
                                             'severity'
                                             ]
                                    )

        error_db.rename(columns={"filename": "path"}, inplace=True)

        file_list = error_db['path'].to_list()

        #########################################
        # Remove the project's root from the path
        common_prefix = os.path.commonprefix(file_list)

        error_db['path'] = error_db['path'].apply(
            lambda _path: os.path.relpath(_path, common_prefix)
        )

        error_db.to_pickle("report.pickle")

        self.params = {}
        aggregated_by_folder = self._aggregate_errors_by_folder(error_db)
        aggregated_by_code = self._aggregate_errors_by_code(error_db)

        plot_id, plot_js = self._construct_plot_js(aggregated_by_folder, "path", "id")
        self.params["id_plot_error_by_folder"] = plot_id
        self.params["js_plot_error_by_folder"] = plot_js

        plot_id, plot_js = self._construct_plot_js(aggregated_by_code, "code", "code")
        self.params["id_plot_error_by_code"] = plot_id
        self.params["js_plot_error_by_code"] = plot_js

        self.write_index()

    def write_index(self):
        report_template = jinja2_env.get_template('dashboard.html')
        rendered = report_template.render(self.params)

        report_filename = "/home/aperez/workspace/python/flake8_reporters/flake8_reporters/templates/test.html"
        with codecs.open(report_filename, 'w', encoding='utf8') as f:
            f.write(rendered)

    def _construct_plot_js(self, aggregated_errors, ids_column, labels_column):

        trace = plotly.graph_objs.Sunburst(
            parents=aggregated_errors['parent'].values,
            values=aggregated_errors['counts'].values,
            ids=aggregated_errors[ids_column].values,
            labels=aggregated_errors[labels_column].values,
            branchvalues='total',
            maxdepth=3,
            marker={"line": {"width": 2}},
        )

        layout = plotly.graph_objs.Layout(
            margin=plotly.graph_objs.layout.Margin(t=0, l=0, r=0, b=0)
        )

        div = plotly.offline.plot(plotly.graph_objs.Figure([trace], layout),
                                  include_plotlyjs=False, output_type='div')

        soup = BeautifulSoup(div, features="html.parser")
        return soup.div.div['id'], jsmin(soup.div.script.text)

    @staticmethod
    def _aggregate_errors_by_folder(error_db):
        """Compute the aggregated statistics by module/directory."""
        file_info = error_db[['path']].drop_duplicates(subset=['path'])
        file_info.set_index('path', inplace=True)

        file_info['counts'] = error_db["path"].value_counts()

        file_info.reset_index(level=0, inplace=True)

        tmp = file_info['path'].apply(os.path.split)
        file_info['parent'] = tmp.apply(lambda x: x[0])
        # Keep the last directory/module as id
        file_info['id'] = tmp.apply(lambda x: x[1])

        total_errors_by_file = error_db["path"].value_counts()

        aggregated_by_folder = total_errors_by_file.copy()

        intermediate_paths = file_info["path"].apply(full_split)

        intermediate_paths = set(
            itertools.chain.from_iterable(intermediate_paths.values)
        )

        for i in intermediate_paths:
            sel = total_errors_by_file.index.get_level_values(0).str.contains(i)
            aggregated_by_folder.loc[i] = total_errors_by_file[sel].sum()

        aggregated_by_folder = pandas.DataFrame(data=aggregated_by_folder.values,
                                                index=aggregated_by_folder.index,
                                                columns=["counts"])

        aggregated_by_folder['path'] = aggregated_by_folder.index

        aggregated_by_folder.reset_index(inplace=True, drop=True)

        aggregated_by_folder = aggregated_by_folder[['path', 'counts']]

        tmp = aggregated_by_folder['path'].apply(os.path.split)
        aggregated_by_folder['parent'] = tmp.apply(lambda x: x[0])
        # Keep the last directory/module as id
        aggregated_by_folder['id'] = tmp.apply(lambda x: x[1])

        sel = aggregated_by_folder['parent'] == ""  # Select root path

        total_error_counts = aggregated_by_folder.loc[sel, 'counts'].sum()

        aggregated_by_folder.loc[sel, 'parent'] = "All"
        aggregated_by_folder.loc[sel, 'id'] = aggregated_by_folder.loc[sel, 'path']

        aggregated_by_folder = pandas.concat(
            [pandas.DataFrame.from_dict({"path": ["All"],
                                         "counts": [total_error_counts],
                                         "parent": [""],
                                         "id": ["All"]}),
             aggregated_by_folder],
            ignore_index=True,
            sort=False)
        return aggregated_by_folder
        # aggregated_by_folder.to_pickle("statistics.pickle")

    @staticmethod
    def _aggregate_errors_by_code(error_db):
        """Compute the aggregated statistics by error code."""

        error_codes = error_db.code.value_counts()
        error_codes = error_codes.reset_index(name="counts")
        error_codes.rename(columns={"index": "code"}, inplace=True)
        parents = error_codes['code'].apply(lambda x: (x[0], x[:2]))
        parents = set(itertools.chain.from_iterable(parents.values))

        aggregated_by_code = pandas.DataFrame(columns=error_codes.columns)
        for i, parent in enumerate(parents):
            sel = error_codes.code.str.contains(parent)
            row = [parent, error_codes[sel]['counts'].sum()]
            aggregated_by_code.loc[i + 1, ['code', 'counts']] = row

        row = ["", error_codes['counts'].sum()]
        aggregated_by_code.loc[0, ['code', 'counts']] = row

        aggregated_by_code = pandas.concat([error_codes, aggregated_by_code],
                                           sort=True).reset_index(drop=True)

        def get_parent(code):
            if len(code) > 2:
                return code[:2]
            elif len(code) == 2:
                return code[0]
            elif len(code) == 1:
                return "All"
            else:
                return ""

        parents = aggregated_by_code['code'].apply(get_parent)
        aggregated_by_code['parent'] = parents

        aggregated_by_code.loc[aggregated_by_code['code'] == "", "code"] = 'All'
        aggregated_by_code.sort_values(['parent'], inplace=True)

        return aggregated_by_code

    @classmethod
    def add_options(cls, options):
        """Add a -- option to the OptionsManager."""

        cls.option_manager = options
        options.add_option(
            '--sunburst',
            help="Generate a report using a sunburst plot",
            default=False,
            action="store_true",
            dest="sunburst"
        )
        options.add_option(
            '--htmldir',
            help="Directory in which to write HTML output.",
            parse_from_config=True,
            default="./flake8_report",
        )
