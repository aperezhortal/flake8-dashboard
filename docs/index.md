flake8-dashboard
================

A flake8 plugin to generate a responsive HTML dashboard summarizing all
the flake8 violations. The resulting dashboard has an easy-to-read
format across a variety of devices and web browsers.

Installation
------------

If flake8 is not installed, run:

``` {.bash}
$ pip install flake8
```

Finally, to install the latest release of the plugin from the Python
Package Index, run:

``` {.bash}
$ pip install flake8-dashboard
```

Alternatively, to install the latest development version (master
branch), run:

``` {.bash}
$ pip install git+https://github.com/aperezhortal/flake8-dashboard
```

Usage
-----

Run flake8 with the `--format=dashboard` option to create a nice-looking
dashboard.

Options:

-   `--outputdir=<output_dir>`: Directory to save the HTML output
    (\"./flake8\_dashboard\" by default).
-   `--debug-info`: Write additional debugging information as csv format
    (flake8 violations and aggregations).
-   `--title=<title>`: Set the dashboard\'s title. No title by default.

Simple usage example:

``` {.bash}
$ flake8 --format=dashboard --outputdir=flake-report --title="My dashboard"
```

### Demo

[Check a demo
here!](https://aperezhortal.github.io/flake8-dashboard/example_dashboard/index.html)

Credits
-------

-   This package was created using the
    [flake8-html](https://github.com/lordmauve/flake8-html) package as a
    template.
-   The dashboard html page was created using the
    [light-bootstrap-dashboard](https://demos.creative-tim.com/light-bootstrap-dashboard/)
    template by [Creative Tim](https://www.creative-tim.com/).
-   The interactive plots are created using [Plotly
    Python](https://plot.ly/python/) .
