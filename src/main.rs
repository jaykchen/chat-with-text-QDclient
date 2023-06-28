use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionResponseMessage, CreateChatCompletionRequestArgs,
    CreateEmbeddingRequestArgs, CreateEmbeddingResponse, Role,
};
use async_openai::Embeddings;
use async_openai::{config::OpenAIConfig, Client};

use anyhow::{anyhow, Error};
use dotenv::dotenv;
use lib::*;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors::VectorsOptions;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CreateCollection, PointId, PointStruct, SearchPoints, Vector, VectorParams, Vectors,
    VectorsConfig,
};

use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::default::Default;
use std::env;
use tch::Device;
use tiktoken_rs::cl100k_base;
use tokio;

static SYSTEM_PROMPT : &str = "Please use the paragraphs of text and code below from the book 'Rust in Action' by Tim McNamara as the context to answer any question.\n\n---\n\n";
static COLLECTION_NAME: &str = "rust_in_action";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let bpe = cl100k_base().unwrap();

    let api_key = env::var("OPENAI_API_TOKEN").unwrap();
    let config = OpenAIConfig::new().with_api_key(api_key);

    let openai_client = Client::with_config(config);

    let config = QdrantClientConfig::from_url("http://127.0.0.1:6334");
    let qdrant_client = QdrantClient::new(Some(config))?;

    // init_collection().await?;

    let model = SentenceEmbeddingsBuilder::local(
        "/Users/jaykchen/Projects/chat-with-text-local/all-MiniLM-L12-v2",
    )
    .with_device(Device::cuda_if_available())
    .create_model()?;
    let s = include_str!("book.txt");
    // println!("segmented_text: {:?}", s);

    let chunked_text = bpe
        .encode_ordinary(s)
        .chunks(3000)
        .map(|c| bpe.decode(c.to_vec()).unwrap())
        .collect::<Vec<String>>();

    // for chunk in chunked_text {
    //     if let Ok(segment) = segment_text(&chunk).await {
    //         upload_embeddings(segment).await?;
    //     }
    // }

    // let segmented_text = vec![
    //     "Why do programmers hate nature? It has too many bugs.",
    //     "Why was the computer cold? It left its Windows open.",
    // ];
    // lib::upsert_points(COLLECTION_NAME, points).await?;
    // let collection_info = client.collection_info(COLLECTION_NAME).await?;
    // dbg!(collection_info);

    let input = "where is Rust used?";
    let input = "how are numbers represented in Rust?";
    let input = "This chapter covers";
    // let request = CreateEmbeddingRequestArgs::default()
    //     .model("text-embedding-ada-002")
    //     .input(input)
    //     .build()
    //     .unwrap();

    // let response: CreateEmbeddingResponse = openai_client.embeddings().create(request).await?;

    let embeddings = model.encode(&[input])?;
    let question_vector = embeddings[0].clone();
    let search_result = qdrant_client
        .search_points(&SearchPoints {
            collection_name: COLLECTION_NAME.into(),
            vector: question_vector,
            filter: None,
            limit: 10,
            with_vectors: None,
            with_payload: None,
            params: None,
            score_threshold: None,
            offset: None,
            ..Default::default()
        })
        .await?;
    dbg!(search_result);

    // println!("{:?}", question_vector[0].clone());
    // let p = PointsSearchParams {
    //     vector: question_vector[0].clone(),
    //     top: 2,
    // };

    // // let mut system_prompt_updated = String::from(SYSTEM_PROMPT);
    // let found_texts = match search_points(COLLECTION_NAME, &p).await {
    //     Ok(sp) => sp
    //         .iter()
    //         .map(|p| {
    //             p.payload
    //                 .as_ref()
    //                 .unwrap()
    //                 .get("text")
    //                 .unwrap()
    //                 .as_str()
    //                 .unwrap()
    //         })
    //         .collect::<Vec<_>>()
    //         .join("\n"),
    //     Err(e) => "".to_string(),
    // };

    // println!("{:?}", found_texts);

    Ok(())
}

fn first_x_chars(s: &str, x: usize) -> String {
    s.chars().take(x).collect()
}

pub async fn init_collection() -> anyhow::Result<()> {
    let config = QdrantClientConfig::from_url("http://127.0.0.1:6334");
    let qdrant_client = QdrantClient::new(Some(config))?;

    qdrant_client
        .create_collection(&CreateCollection {
            collection_name: COLLECTION_NAME.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 384,
                    distance: Distance::Cosine.into(),
                    hnsw_config: None,
                    quantization_config: None,
                    on_disk: None,
                })),
            }),
            ..Default::default()
        })
        .await?;

    Ok(())
}

pub async fn upload_embeddings(inp: Vec<String>) -> anyhow::Result<()> {
    // let api_key = env::var("OPENAI_API_TOKEN").unwrap();
    // let config = OpenAIConfig::new().with_api_key(api_key);

    // let openai_client = Client::with_config(config);
    let model = SentenceEmbeddingsBuilder::local(
        "/Users/jaykchen/Projects/chat-with-text-local/all-MiniLM-L12-v2",
    )
    .with_device(Device::cuda_if_available())
    .create_model()?;

    let config = QdrantClientConfig::from_url("http://127.0.0.1:6334");
    let qdrant_client = QdrantClient::new(Some(config))?;

    // let request = CreateEmbeddingRequestArgs::default()
    //     .model("text-embedding-ada-002")
    //     .input(&inp)
    //     .build()?;

    // let response: CreateEmbeddingResponse = openai_client.embeddings().create(request).await?;

    // let embeddings = response.data;

    let embeddings = model.encode(&inp)?;

    for data in embeddings.clone() {
        println!(" has embedding of length {}", embeddings.len())
    }

    // let sentences_stemmed = s.split(",").collect::<Vec<_>>();
    // // println!("{:?}", sentences_stemmed[0]);

    // let collection_params = CollectionCreateParams {
    //     vectors: VectorParams {
    //         size: 384,
    //         distance: "Dot".to_string(),
    //     },
    //     optimizers_config: OptimizersConfig {
    //         default_segment_number: 4,
    //     },
    //     replication_factor: 2,
    // };

    // create_collection(COLLECTION_NAME, &collection_params).await?;

    let mut points = Vec::new();
    for (i, sentence) in inp.iter().enumerate() {
        let payload: Payload = serde_json::json!({ "text": sentence.trim().to_string()})
            .try_into()
            .unwrap();
        let point = PointStruct::new(
            PointId {
                point_id_options: Some(PointIdOptions::Num(i as u64)),
            },
            Vectors {
                vectors_options: Some(VectorsOptions::Vector(Vector {
                    data: embeddings[i].clone(),
                })),
            },
            payload,
        );

        // let point = Point {
        //     id: PointId::Num(i as u64),
        //     vector: embeddings[i].clone(),
        //     payload: Some(
        //         vec![(
        //             "text".to_string(),
        //             serde_json::json!(sentence.trim().to_string()),
        //         )]
        //         .into_iter()
        //         .collect::<Map<String, Value>>(),
        //     ),
        // };
        // println!("{:?}", point.vector[0]);

        points.push(point);
    }
    println!("{:?}", points.len());

    qdrant_client
        .upsert_points_blocking(COLLECTION_NAME, points, None)
        .await?;
    Ok(())
}
pub async fn segment_text(inp: &str) -> anyhow::Result<Vec<String>> {
    let api_key = env::var("OPENAI_API_TOKEN").unwrap();
    let config = OpenAIConfig::new().with_api_key(api_key);

    let openai_client = Client::with_config(config);

    let prompt = format!(
        r#"You are examining Chapter 1 of a book. Your mission is to dissect the provided information into short, logically divided segments to facilitate further processing afterwards. 
    Please adhere to the following steps:
    1. Break down dense paragraphs into individual sentences, with each one functioning as a distinct chunk of information. Use the marker "---" to indicate the end of each segment.
    2. Consider code snippets as standalone entities and separate them from the accompanying text. Utilize the marker "---" to indicate the conclusion of each code snippet.
    3. Take into account the original source's hierarchical markings and formatting specific to a book chapter. These elements can guide the logical segmentation process.
    Keep in mind, the goal is not to summarize, but to restructure the information into more digestible, manageable units.
    Now, here is the text from the chapter:{inp}"#
    );
    let system_message = ChatCompletionRequestMessage {
        role: Role::System,
        content: Some(r#"As a dedicated assistant, your duty is to dissect the provided chapter text into clearer, bite-sized segments. To accomplish this, isolate each sentence and code snippet as independent entities, each delineated by a "---". Remember, your task is not to provide a summary, but to split the original text into a texts sequence more granunlar, respecting the text's hierarchical markings and formatting as they contribute to the understanding of the text. Balance your interpretations with the original structure for an accurate representation."#.to_string()),
        name: None,
        function_call: None,
};

    let user_message = ChatCompletionRequestMessage {
        role: Role::User,
        content: Some(prompt),
        name: None,
        function_call: None,
    };

    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-3.5-turbo-16k")
        .messages(vec![system_message, user_message])
        .max_tokens(7000_u16)
        .build()?;

    let response = openai_client
        .chat() // Get the API "group" (completions, images, etc.) from the client
        .create(request) // Make the API call in that "group"
        .await?;

    match response.choices[0].message.content.clone() {
        Some(raw_text) => Ok(raw_text
            .split("---")
            .map(|x| x.to_string())
            .collect::<Vec<_>>()),
        None => Err(anyhow::anyhow!("Could not get the text from OpenAI")),
    }
}
