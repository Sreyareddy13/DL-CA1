using Flux, Images, ImageIO, FileIO, Statistics, Printf
using Flux: onehotbatch, onecold, params

using BSON: @save, @load

# Load dataset function
function load_dataset(path::String, size::Tuple{Int64, Int64} = (224, 224))
    println("✅ Checking dataset folder: $path")
    class_dirs = filter(x -> isdir(joinpath(path, x)), readdir(path))
    sort!(class_dirs)
    println("✅ Classes found: ", class_dirs)

    X = Array{Float32, 4}[]
    y = Int64[]
    
    for (i, class) in enumerate(class_dirs)
        class_dir = joinpath(path, class)
        imgs = filter(x -> endswith(x, ".jpg") || endswith(x, ".JPG") || endswith(x, ".png"), readdir(class_dir))
        imgs = first(imgs, min(length(imgs), 100))  # limit to 100 images per class
        println("📁 $class -> $(length(imgs)) images")

        for img in imgs
            try
                img_path = joinpath(class_dir, img)
                image = load(img_path)
                image_resized = imresize(image, size)
                image_array = permutedims(channelview(image_resized), (3, 2, 1))
                push!(X, convert(Array{Float32, 3}, image_array))
                push!(y, i)
            catch e
                @warn "❌ Skipping $img due to error: $e"
            end
        end
    end

    # Stack images into 4D tensor
    X_tensor = cat(X..., dims=4)
    y_encoded = onehotbatch(y, 1:length(class_dirs))

    return X_tensor, y_encoded, length(class_dirs)
end

# Load your PlantVillage dataset
X, y, num_classes = load_dataset("C:\\Users\\sreya\\OneDrive\\Desktop\\dl_crop_detection\\PlantVillage")

# Define the model (AlexNet-like)
model = Chain(
    Conv((11,11), 3=>64, relu; stride=4, pad=2),
    MaxPool((3,3); stride=2),
    Conv((5,5), 64=>192, relu; pad=2),
    MaxPool((3,3); stride=2),
    Conv((3,3), 192=>384, relu; pad=1),
    Conv((3,3), 384=>256, relu; pad=1),
    Conv((3,3), 256=>256, relu; pad=1),
    MaxPool((3,3); stride=2),
    x -> reshape(x, :, size(x, 4)),
    Dense(256 * 6 * 6, 4096, relu),
    Dropout(0.5),
    Dense(4096, 4096, relu),
    Dropout(0.5),
    Dense(4096, num_classes),
    softmax
)
using Random
Random.seed!(42)  # For reproducibility

# Shuffle indices
n = size(X, 4)
indices = shuffle(1:n)
train_size = Int(round(0.8 * n))
train_idx = indices[1:train_size]
test_idx = indices[train_size+1:end]

# Split X and y
X_train = X[:, :, :, train_idx]
y_train = y[:, train_idx]
X_test = X[:, :, :, test_idx]
y_test = y[:, test_idx]

loss(x, y) = Flux.crossentropy(model(x), y)
opt = ADAM()

# Train for 3 epochs
for epoch in 1:3
    @info "🔁 Epoch $epoch"
    grads = gradient(Flux.params(model)) do
        loss(X, y)
    end
    Flux.Optimise.update!(opt, Flux.params(model), grads)
end

# Prediction and evaluation


# Compute and print accuracy
# Compute and print accuracy
ŷ = model(X_test)
predictions = onecold(ŷ, 1:num_classes)
truth = onecold(y_test, 1:num_classes)
accuracy = mean(predictions .== truth)

println("✅ Final Accuracy: ", round(accuracy * 100, digits=2), "%")





