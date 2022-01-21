FROM rust:latest

WORKDIR /usr/src/app
COPY . .

EXPOSE 2000

RUN cargo install --path .
RUN cargo build --release
CMD cargo run