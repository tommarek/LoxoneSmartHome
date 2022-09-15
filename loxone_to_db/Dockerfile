FROM rust:latest

WORKDIR /usr/src/app
COPY . .

EXPOSE 2000/udp
RUN chmod -R 777 /usr/local/cargo
RUN cargo install --path .
RUN cargo build --release
CMD cargo run
