use http_req::{
    request::{Method, Request},
    uri::Uri,
};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

lazy_static! {
    static ref VECTOR_STORE_API_PREFIX: String = String::from(
        std::option_env!("VECTOR_STORE_API_PREFIX").unwrap_or("http://127.0.0.1:6333/collections")
    );
}

/// The information of the collection.
/// A collection is a named set of points (vectors with a payload) among which you can search.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectionInfo {
    /// Count of points in the collection
    pub points_count: u64,
}

/// The parameters for creating the collection
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorParams {
    pub size: u64,
    pub distance: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizersConfig {
    pub default_segment_number: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectionCreateParams {
    pub vectors: VectorParams,
    pub optimizers_config: OptimizersConfig,
    pub replication_factor: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PointId {
    Uuid(String),
    Num(u64),
}

/// The point struct.
/// A point is a record consisting of a vector and an optional payload.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Point {
    /// Id of the point
    pub id: PointId,

    /// Vectors
    pub vector: Vec<f32>,

    /// Additional information along with vectors
    pub payload: Option<Map<String, Value>>,
}

/// The parameters for searching for points
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PointsSearchParams {
    /// Vectors
    pub vector: Vec<f32>,

    /// Max number of result to return
    pub top: u64,
}

/// The point struct with the score returned by searching
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScoredPoint {
    /// Id of the point
    pub id: PointId,

    /// Vectors
    pub vector: Option<Vec<f32>>,

    /// Additional information along with vectors
    pub payload: Option<Map<String, Value>>,

    /// Points vector distance to the query vector
    pub score: f32,
}

/// Get detailed information about specified existing collection.
pub async fn collection_info(collection_name: &str) -> Result<CollectionInfo, String> {
    let mut writer = Vec::new();
    let uri = format!("{}/{}", VECTOR_STORE_API_PREFIX.as_str(), collection_name,);
    let uri = Uri::try_from(uri.as_str()).unwrap();
    match Request::new(&uri)
        .method(Method::GET)
        .header("Content-Type", "application/json")
        .send(&mut writer)
    {
        Ok(res) => {
            if res.status_code().is_success() {
                serde_json::from_slice::<CollectionInfo>(&writer).or_else(|e| Err(e.to_string()))
            } else {
                let err = String::from_utf8_lossy(&writer);
                log::error!("{}", err);
                Err(err.into_owned())
            }
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Create new collection with given parameters
pub async fn create_collection(
    collection_name: &str,
    params: &CollectionCreateParams,
) -> Result<(), String> {
    let mut writer = Vec::new();
    let uri = format!("{}/{}", VECTOR_STORE_API_PREFIX.as_str(), collection_name,);
    let uri = Uri::try_from(uri.as_str()).unwrap();
    let body = serde_json::to_vec(&params).unwrap_or_default();

    match Request::new(&uri)
        .method(Method::PUT)
        .header("Content-Type", "application/json")
        .header("Content-Length", &body.len())
        .body(&body)
        .send(&mut writer)
    {
        Ok(res) => {
            if res.status_code().is_success() {
                Ok(())
            } else {
                let err = String::from_utf8_lossy(&writer);
                log::error!("{}", err);
                Err(err.into_owned())
            }
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Drop collection and all associated data
pub async fn delete_collection(collection_name: &str) -> Result<(), String> {
    let mut writer = Vec::new();
    let uri = format!("{}/{}", VECTOR_STORE_API_PREFIX.as_str(), collection_name,);
    let uri = Uri::try_from(uri.as_str()).unwrap();
    match Request::new(&uri)
        .method(Method::DELETE)
        .header("Content-Type", "application/json")
        .send(&mut writer)
    {
        Ok(res) => {
            if res.status_code().is_success() {
                Ok(())
            } else {
                let err = String::from_utf8_lossy(&writer);
                log::error!("{}", err);
                Err(err.into_owned())
            }
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Perform insert + updates on points. If point with given ID already exists - it will be overwritten.
pub async fn upsert_points(collection_name: &str, points: Vec<Point>) -> Result<(), String> {
    let mut writer = Vec::new();
    let uri = format!(
        "{}/{}/points?wait=true",
        VECTOR_STORE_API_PREFIX.as_str(),
        collection_name,
    );
    let uri = Uri::try_from(uri.as_str()).unwrap();
    let points = serde_json::json!({ "points": points });
    let body = serde_json::to_vec(&points).unwrap_or_default();

    match Request::new(&uri)
        .method(Method::PUT)
        .header("Content-Type", "application/json")
        .header("Content-Length", &body.len())
        .body(&body)
        .send(&mut writer)
    {
        Ok(res) => {
            if res.status_code().is_success() {
                Ok(())
            } else {
                let err = String::from_utf8_lossy(&writer);
                log::error!("{}", err);
                Err(err.into_owned())
            }
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Retrieve closest points based on vector similarity and given filtering conditions
pub async fn search_points(
    collection_name: &str,
    params: &PointsSearchParams,
) -> Result<Vec<ScoredPoint>, String> {
    let mut writer = Vec::new();
    let uri = format!(
        "{}/{}/points/search",
        VECTOR_STORE_API_PREFIX.as_str(),
        collection_name,
    );
    let uri = Uri::try_from(uri.as_str()).unwrap();
    let body = serde_json::to_vec(&params).unwrap_or_default();

    match Request::new(&uri)
        .method(Method::POST)
        .header("Content-Type", "application/json")
        .header("Content-Length", &body.len())
        .body(&body)
        .send(&mut writer)
    {
        Ok(res) => {
            if res.status_code().is_success() {
                serde_json::from_slice::<Vec<ScoredPoint>>(&writer).or_else(|e| Err(e.to_string()))
            } else {
                let err = String::from_utf8_lossy(&writer);
                log::error!("{}", err);
                Err(err.into_owned())
            }
        }
        Err(e) => Err(e.to_string()),
    }
}
