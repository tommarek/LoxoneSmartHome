extern crate chrono;
extern crate mongodb;
extern crate serde;
extern crate tokio;

use mongodb::{ Client, options::ClientOptions };
use std::net::UdpSocket;
use std::{ str, env, error };
use chrono::serde::ts_seconds;
use chrono::{ offset::TimeZone, DateTime, NaiveDateTime, Utc };
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct LogLine {
    #[serde(with = "ts_seconds")]
	time: DateTime<Utc>,
	measurement_name: String,
	value: f32,
	alias: String,
	tag1: String,
	tag2: String,
	tag3: String,
}

// boxed errors to allow mutliple erros in a fucntion
// https://doc.rust-lang.org/rust-by-example/error/multiple_error_types/boxing_errors.html
type MyResult<T> = std::result::Result<T, Box<dyn error::Error>>;


fn parse_data(received_data: &str) -> MyResult<LogLine> {
	let values: Vec<&str> = received_data.split(';').collect();
	let (mut alias, mut tag1, mut tag2, mut tag3) = (String::new(), String::new(), String::new(), String::new());

	if values.len() >= 4 {
		alias = (&values[3]).to_string();
	}
	if values.len() >= 5 {
		tag1 = (&values[4]).to_string();
	}
	if values.len() >= 6 {
		tag2 = (&values[5]).to_string();
	}
	if values.len() >= 7 {
		tag3 = (&values[6]).to_string();
	}

	// Convert native datetime to local timezone
	let naive_datetime = NaiveDateTime::parse_from_str(&values[0], "%Y-%m-%d %H:%M:%S")?;
	let local_datetime: DateTime<Utc> = Utc.from_local_datetime(&naive_datetime).unwrap();

	let log_line = LogLine {
		time: local_datetime,
		measurement_name: (&values[1]).to_string(),
		value: (&values[2]).to_string().parse::<f32>()?,
		alias: alias,
		tag1: tag1,
		tag2: tag2,
		tag3: tag3
	};

	Ok(log_line)
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
	let socket = UdpSocket::bind("0.0.0.0:2000")?;
	let mut buf = [0; 2048];
	
	let db_uri = env::var("MONGODB_URI")?;
	let db_name = env::var("DB")?;

	let client_options = ClientOptions::parse(db_uri).await?;
	let client = Client::with_options(client_options)?;
	let db = client.database(db_name.as_str());
	let typed_collection = db.collection::<LogLine>("loxone");

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
			log_line.time,
			log_line.measurement_name,
			log_line.value,
			log_line.alias,
			log_line.tag1,
			log_line.tag2,
			log_line.tag3
		);

		typed_collection.insert_one(log_line, None).await?;

	}
}
