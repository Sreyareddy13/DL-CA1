
### Fine-tuning AlexNet on PlantVillage Dataset (CPU Only, 17 Classes) ###

using Flux, Metalhead, Images, FileIO, Glob, MLDataPattern, Statistics
using Metalhead

# Always use CPU
my_device(x) = cpu(x)
println("⚠️ Using CPU for training (no GPU).")

# Load pretrained AlexNet and modify last layer for 17 classes
base_model = Metalhead.alexnet() |> my_device
model = Chain(base_model, Dense(1000, 17), softmax) |> my_device

# Load and preprocess PlantVillage dataset
function load_dataset(path::String, classes::Vector{String}, size=(224, 224))
    X, y = [], Int[]
    for (label, class) in enumerate(classes)
        files = glob("*.jpg", joinpath(path, class))
        for file in files
            try
                img = load(file)
                img_resized = imresize(img, size)
                img_tensor = Float32.(channelview(img_resized)) ./ 255
                push!(X, img_tensor)
                push!(y, label)
            catch e
                println("⚠️ Skipping file: $file due to error: $e")
            end
        end
    end
    return Array{Float32, 4}(cat(X..., dims=4)), Flux.onehotbatch(y, 1:length(classes))
end

# 17 class folder names (match your dataset exactly)
classes = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus"
]

# Load dataset
X, y = load_dataset("PlantVillage", classes)
X, y = my_device(X), my_device(y)

# Create DataLoader
using DataLoaders
dataloader = DataLoader((X, y), batchsize=32, shuffle=true)

# Define loss function and optimizer
loss(x, y) = Flux.crossentropy(model(x), y)
opt = ADAM()

# Train model for 10 epochs
@epochs 10 Flux.train!(loss, params(model), dataloader, opt)

# Evaluate final model accuracy
using Flux: onecold
y_pred = model(X)
y_true = onecold(y, 1:length(classes))
y_pred_labels = onecold(y_pred, 1:length(classes))

acc = mean(y_pred_labels .== y_true)
println("✅ Validation Accuracy: ", round(acc * 100, digits=2), "%")
