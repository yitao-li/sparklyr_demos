library(sparklyr)

sc <- spark_connect(master = "local", version = "3.0.0")

path <- tempfile("streaming_demo_")
dir.create(path, recursive = TRUE)
# NOTE: `stream_generate_test()` requires `callr` package to be installed
stream_generate_test(
  tibble::tibble(x = rnorm(100L)),
  distribution = 10L + floor(1e+04 * stats::dbinom(1:20, 20, 0.5)),
  iterations = 10000L,
  path = path,
  interval = 0.1
)

sdf <- stream_read_csv(sc, path = sprintf("file://%s", path))
print(sdf %>% sdf_is_streaming())

stream <- sdf %>%                       # source
  dplyr::mutate(x = x * x) %>%          # some intermediate transformations
  dplyr::filter(x < 2) %>%              # some intermediate filtering
  stream_write_memory(mode = "update")  # sink

stream_view(stream)
