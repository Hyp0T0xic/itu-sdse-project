package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

func main() {
	if err := build(context.Background()); err != nil {
		fmt.Println(err)
		panic(err)
	}
}

func build(ctx context.Context) error {
	fmt.Println("Building with Dagger...")

	// 1. Connect to Dagger
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
	if err != nil {
		return err
	}
	defer client.Close()

	// 2. Define the container
	// We use a standard Python image
	container := client.Container().
		From("python:3.11-slim").
		// Mount the source code from the host (current directory) to /app in the container
		// "." refers to the directory where 'dagger run' is executed (project root)
		WithDirectory("/app", client.Host().Directory("..")).
		WithWorkdir("/app").
		// Install dependencies
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		// Run the training pipeline
		WithExec([]string{"python", "-m", "src.main"})

	// 3. Export the artifacts
	// The pipeline produces 'artifacts/model.pkl', we want to save it back to our host
	_, err = container.Directory("artifacts").Export(ctx, "../artifacts")
	if err != nil {
		return err
	}

	// Export MLflow runs
	_, err = container.Directory("mlruns").Export(ctx, "../mlruns")
	if err != nil {
		return err
	}

	fmt.Println("Pipeline finished! Model exported to ./artifacts")
	return nil
}
