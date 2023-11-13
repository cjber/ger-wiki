FROM cjber/cuda:0.1

ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

ENV PYTHON_VERSION=3.8.5
ENV POETRY_VERSION=1.1.14

WORKDIR /ger_wiki
COPY . .

RUN --mount=type=cache,sharing=locked,target=/var/cache/pacman \
    pacman -Syu --noconfirm pyenv r blas lapack gcc-fortran --noconfirm \
    && yes | pacman -Scc --noconfirm \
    && pyenv install "${PYTHON_VERSION}" \
    && pyenv global "${PYTHON_VERSION}"  \
    && pyenv rehash \
    && pip install --no-cache-dir --upgrade pip==22.2 poetry=="${POETRY_VERSION}" \
    && poetry install \
    && yes | poetry cache clear pypi --all
