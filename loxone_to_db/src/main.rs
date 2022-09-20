extern crate chrono;
extern crate tokio;
extern crate influxrs;

use chrono::{TimeZone, NaiveDateTime};
use chrono_tz::Europe::Prague;
use chrono_tz::UTC;
use std::net::UdpSocket;
use std::{ str, error };
use influxrs::{ Measurement, InfluxClient};


struct LogLine {
	timestamp: u128,
	measurement_name: String,
	value: f64,
	room: String,
	measurement_type: String,
	tag1: String,
	tag2: String,
}

// boxed errors to allow mutliple erros in a fucntion
// https://doc.rust-lang.org/rust-by-example/error/multiple_error_types/boxing_errors.html
type MyResult<T> = std::result::Result<T, Box<dyn error::Error>>;


// data structure:
// timestamp;measurement_name;value;room_name[optional];measurement_type[optional];tag1[optional];tag2[optional]
fn parse_data(received_data: &str) -> MyResult<LogLine> {
	let values: Vec<&str> = received_data.split(';').collect();
	let (mut room, mut measurement_type, mut tag1, mut tag2) = ("_".to_string(), "default".to_string(), "_".to_string(), "_".to_string());

	if values.len() >= 4 {
		room = (&values[3]).to_string();
	}
	if values.len() >= 5 {
		measurement_type = (&values[4]).trim().to_string();
	}
	if values.len() >= 6 {
		tag1 = (&values[5]).to_string();
	}
	if values.len() >= 7 {
		tag2 = (&values[6]).to_string();
	}

	// Loxone sends timestamps in Local time, we need to convert it to UTC
	let naive_datetime = NaiveDateTime::parse_from_str(&values[0], "%Y-%m-%d %H:%M:%S").unwrap();
	let tz_aware_datetime = Prague.from_local_datetime(&naive_datetime).unwrap();
	let utc_datetime = tz_aware_datetime.with_timezone(&UTC);
	let timestamp: u128 = utc_datetime.timestamp_millis().try_into().unwrap();

	let log_line = LogLine {
		timestamp: timestamp,
		measurement_name: (&values[1]).replace(" ", "_").to_lowercase().to_string(),
		value: (&values[2]).to_string().parse::<f64>()?,
		measurement_type: measurement_type,
		room: room,
		tag1: tag1,
		tag2: tag2
	};

	Ok(log_line)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
	let socket = UdpSocket::bind("0.0.0.0:2000")?;
	let mut buf = [0; 2048];

	let host = std::env::var("INFLUXDB_HOST").unwrap();
	let org = std::env::var("INFLUXDB_ORG").unwrap();
	let token = std::env::var("INFLUXDB_TOKEN").unwrap();
	let bucket = std::env::var("INFLUXDB_BUCKET").unwrap();

	// let host = "http://localhost:8086";
	// let org = "loxone";
	// let bucket = "loxone";
	// let token = "0YNpYaSILBpQMxXemDuqiwNgOMVwT-pDyOz8F1-DFh8yVo-ntOSlpzEmg6qYBX356jBKrSYkrxxt5msrx-lLBw==";

	let client = InfluxClient::builder(host.to_string(), token.to_string(), org.to_string()).build().unwrap();

	println!("Starting to accept data.");
	loop {
		let (amt, _src) = match socket.recv_from(&mut buf) {
			Ok((amt, src)) => (amt, src),
			Err(err) => {
				eprintln!("Failed to receive a message: {}", err);
				continue;
			}
		};

		let buf = &mut buf[..amt];
		let log_line = match parse_data(str::from_utf8(&buf).unwrap()){
			Ok(log_line) => log_line,
			Err(err) => {
				eprintln!("Falied to parse incoming data: {}", err);
				continue;
			}
		};


		println!("{:?};{:?};{:?};{:?};{:?};{:?};{:?}",
			log_line.timestamp,
			log_line.measurement_name,
			log_line.value,
			log_line.room,
			log_line.measurement_type,
			log_line.tag1,
			log_line.tag2
		);

		let measurement = Measurement::builder(log_line.measurement_type)
				.timestamp_ms(log_line.timestamp)
				.field(log_line.measurement_name, log_line.value)
				.tag("room", log_line.room)
				.tag("tag1", log_line.tag1)
				.tag("tag2", log_line.tag2)
				.build()
				.unwrap();

		let _response = match client
			.write(bucket.as_str(), &[measurement])
			.await {
				Ok(response) => {
					println!("Stored new log line.");
					response
				},
				Err(err) => {
					eprintln!("Falied to store incoming data: {}", err);
					continue;
				}
			};
	}

}
