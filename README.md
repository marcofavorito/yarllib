<h1 align="center">
    <b>yarllib</b>
</h1>

<p align="center">
  <a href="https://github.com/marcofavorito/yarllib/actions?query=workflow%3Atest">
    <img alt="test" src="https://github.com/marcofavorito/yarllib/workflows/test/badge.svg" />
  </a>
  <a href="https://github.com/marcofavorito/yarllib/actions?query=workflow%3Alint">
    <img alt="lint" src="https://github.com/marcofavorito/yarllib/workflows/lint/badge.svg" />
  </a>
  <a href="https://github.com/marcofavorito/yarllib/actions?query=workflow%3Adocs">
    <img alt="docs" src="https://github.com/marcofavorito/yarllib/workflows/docs/badge.svg" />
  </a>
  <a href="https://codecov.io/gh/marcofavorito/yarllib">
    <img alt="Coverage" src="https://codecov.io/gh/marcofavorito/yarllib/branch/master/graph/badge.svg?token=RFZ54P0BKQ" />
  </a>
  <a href="https://github.com/marcofavorito/yarllib/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/marcofavorito/yarrlib">
  </a>
</p>
<p align="center">
  <a href="https://img.shields.io/badge/flake8-checked-blueviolet">
    <img alt="flake8" src="https://img.shields.io/badge/flake8-checked-blueviolet" />
  </a>
  <a href="https://img.shields.io/badge/mypy-checked-blue">
    <img alt="mypy" src="https://img.shields.io/badge/mypy-checked-blue" />
  </a>
  <a href="https://img.shields.io/badge/isort-checked-yellow">
    <img alt="isort" src="https://img.shields.io/badge/isort-checked-yellow" />
  </a>
  <a href="https://img.shields.io/badge/code%20style-black-black">
    <img alt="black" src="https://img.shields.io/badge/code%20style-black-black" />
  </a>
  <a href="https://img.shields.io/badge/docs-mkdocs-9cf">
    <img alt="mkdocs" src="https://img.shields.io/badge/docs-mkdocs-9cf" />
  </a>
</p>

Yet Another Reinforcement Learning Library.

Status: **development**.

## Tests

To run tests: `tox`

To run only the code tests: `tox -e py3.7`

To run only the linters: 
- `tox -e flake8`
- `tox -e mypy`
- `tox -e black-check`
- `tox -e isort-check`

Please look at the `tox.ini` file for the full list of supported commands. 

## Docs

To build the docs: `mkdocs build`

To view documentation in a browser: `mkdocs serve`
and then go to [http://localhost:8000](http://localhost:8000)

## License

yarllib is released under the GNU Lesser General Public License v3.0 or later (LGPLv3+).

Copyright 2020 Marco Favorito

## Authors

- [Marco Favorito](https://marcofavorito.github.io/)