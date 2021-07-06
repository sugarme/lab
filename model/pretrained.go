package model

import (
	"fmt"
)

type resource struct{
	file string
	url string
}

// Pretrained
var PretrainedModels map[string]resource = map[string]resource{
	"efficientnet_b4": {
		file: "efficientnet_b4.bin",
		url: "https://PRETRAINED_MODEL_FILES",
	},
	"efficientnet_b6": {
		file: "efficientnet_b6.bin",
		url: "https://PRETRAINED_MODEL_FILES",
	},
	"efficientnet_b7": {
		file: "efficientnet_b7.bin",
		url: "https://PRETRAINED_MODEL_FILES",
	},


	"tf_efficientnet_b6_ns": {
		file: "tf_efficientnet_b6_ns.bin",
		url: "https://PRETRAINED_MODEL_FILES",
	},
	"tf_efficientnet_b7_ns": {
		file: "tf_efficientnet_b7_ns.bin",
		url: "https://PRETRAINED_MODEL_FILES",
	},
}

type PretrainedOptions struct{
	Path string
	URL string
	CacheDir string
}

type PretrainedOption func(*PretrainedOptions)

func defaultPretrainedOptions() *PretrainedOptions{
	return &PretrainedOptions{
		Path: "",
		URL: "",
		CacheDir: DefaultCachePath,
	}
}

// WithPath set path to model file
func WithPath(p string) PretrainedOption{
	return func(o *PretrainedOptions){
		o.Path = p
	}
}

// WithURL set URL to retrieve model file from remote resource.
func WithURL(url string) PretrainedOption{
	return func(o *PretrainedOptions){
		o.URL = url
	}
}

// WithCacheDir set custom cache directory to cache model.
// Default cache directory is at `$HOME/.cache/lab`
func WithCacheDir(dir string) PretrainedOption{
	return func(o *PretrainedOptions){
		o.CacheDir = dir
	}
}

// PretrainedFile returns corresponding pretrained model file from cached file.
// If cached file does not exist it will download from resource first.
func PretrainedFile(modelName string, opts ...PretrainedOption) (string, error){
	options := defaultPretrainedOptions()
	for _, o := range opts{
		o(options)
	}

	DefaultCachePath = options.CacheDir
	resource, ok := PretrainedModels[modelName] 
	if !ok{
		err := fmt.Errorf("Unsupported model name: %s\n", modelName)
		return "", err
	}

	// For now just load from local
	fullPath := fmt.Sprintf("%s/%s", options.Path, resource.file)
	return CachedPath(fullPath)
}

