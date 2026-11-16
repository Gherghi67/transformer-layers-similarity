FROM python:3.14-slim-bookworm@sha256:3fe1a0b6c2d6b2ebb572906cdabeca75aeb558c0c17db5ffefbcf8094871c32b

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN groupadd --gid 10001 atlas && useradd --uid 10001 --gid 10001 --no-create-home atlas \
    && mkdir /data && chown 10001:10001 /data && chmod 700 /data
WORKDIR /app
COPY --chown=root:root server.py ./
COPY --chown=root:root static/ ./static/
COPY --chown=root:root examples/example.json ./examples/example.json
RUN chmod -R a+rX /app
USER 10001:10001
EXPOSE 8786
CMD ["python", "server.py", "--bind", "0.0.0.0", "--port", "8786", "--db", "/data/layer-atlas.sqlite3", "--trust-proxy-identity", "--allowed-host", "macbook-pro-2.tailaedfae.ts.net"]
