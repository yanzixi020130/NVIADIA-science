# NVIDIA Science Data Processing

本项目提供数据上传预览、点云/mesh 可视化、STL 几何采样，以及基于 `physicsnemo.mesh` 的 mesh 分析和物理场分析接口。

当前重点接口都挂在 `/data-processing` 下：

- `GET /data-processing/physicsnemo/status`
- `POST /data-processing/upload-preview`
- `POST /data-processing/geometry/sample`
- `POST /data-processing/mesh/analyze`
- `POST /data-processing/mesh/field-analysis`

## 环境准备

推荐使用 conda 环境，Python 版本使用 3.10。

```bash
conda create -n AnalysisData python=3.10 -y
conda activate AnalysisData
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

CUDA 12.x 服务器上，`requirements.txt` 已包含：

```txt
nvidia-physicsnemo[cu12,mesh-extras,sym]
```

确认 PyTorch/CUDA：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 启动服务

默认端口是 `1101`。

```bash
conda activate AnalysisData
python main.py
```

另一个终端中设置测试变量：

```bash
export BASE_URL=http://localhost:1101
export PROJECT=$(pwd)
```

## 测试数据

新增测试数据位于：

```text
test_data/physicsnemo_mesh_demo/
```

文件说明：

- `plane_temperature_velocity.csv`：规则平面点云，包含 `temperature`、`pressure`、`u/v/w`。
- `wavy_surface_fields.csv`：轻微起伏曲面点云，包含 `temperature`、`stress`、`u/v/w`。
- `square_field.vtp`：两三角面组成的方形 mesh，带点数据和 cell 数据。
- `tetrahedron_closed.stl`：闭合四面体 STL，用于几何采样和 mesh 分析。
- `open_square.obj`：开放方形面 OBJ，用于验证非闭合 mesh 分析。

## 接口测试

### 1. PhysicsNeMo 状态检查

检查 `physicsnemo`、`physicsnemo.sym`、`physicsnemo.mesh` 是否可用。

```bash
curl -s "$BASE_URL/data-processing/physicsnemo/status" | python -m json.tool
```

期望看到：

```json
{
  "physicsnemo_available": true,
  "physicsnemo_mesh_available": true
}
```

### 2. 上传 CSV 并生成可视化和 mesh 分析

`upload-preview` 会保存上传文件、识别字段、生成点云/mesh 可视化。选择 `delaunay_2d` 或 `surface_reconstruction` 时，会自动尝试返回 `mesh_analysis`。

```bash
curl -s -X POST "$BASE_URL/data-processing/upload-preview" \
  -F "taskid=mesh-demo" \
  -F "file=@$PROJECT/test_data/physicsnemo_mesh_demo/plane_temperature_velocity.csv" \
  -F "coord_cols=x,y,z" \
  -F "color_col=temperature" \
  -F "vector_cols=u,v,w" \
  -F "visualization_mode=delaunay_2d" \
  -F "projection_plane=xy" \
  -F "show_grid=true" \
  -F "mesh_analysis_device=cuda:0" \
  | python -m json.tool
```

如果 GPU 正忙，可以改成 CPU：

```bash
-F "mesh_analysis_device=cpu"
```

只做上传和可视化，不做 PhysicsNeMo 分析：

```bash
curl -s -X POST "$BASE_URL/data-processing/upload-preview" \
  -F "taskid=mesh-demo" \
  -F "file=@$PROJECT/test_data/physicsnemo_mesh_demo/wavy_surface_fields.csv" \
  -F "coord_cols=x,y,z" \
  -F "color_col=temperature" \
  -F "visualization_mode=surface_reconstruction" \
  -F "show_grid=true" \
  -F "include_mesh_analysis=false" \
  | python -m json.tool
```

### 3. STL 几何采样

`geometry/sample` 用于 STL 几何边界和内部点采样。接口会优先尝试 `physicsnemo.sym.geometry.tessellation_warp.Tessellation`，失败时 fallback 到本地 `src.geometry.Tessellation`。

```bash
curl -s -X POST "$BASE_URL/data-processing/geometry/sample" \
  -H "Content-Type: application/json" \
  -d "{
    \"path\": \"$PROJECT/test_data/physicsnemo_mesh_demo/tetrahedron_closed.stl\",
    \"sample_points\": 200,
    \"return_points\": 10,
    \"include_boundary\": true,
    \"include_interior\": true,
    \"compute_sdf_derivatives\": true,
    \"device\": \"cuda:0\"
  }" \
  | python -m json.tool
```

### 4. Mesh 几何分析

`mesh/analyze` 支持 `.vtp`、`.vtk`、`.stl`、`.ply`、`.obj`，也支持 CSV 点云先重建成三角网格再分析。

分析带场数据的 VTP：

```bash
curl -s -X POST "$BASE_URL/data-processing/mesh/analyze" \
  -H "Content-Type: application/json" \
  -d "{
    \"path\": \"$PROJECT/test_data/physicsnemo_mesh_demo/square_field.vtp\",
    \"device\": \"cuda:0\",
    \"include_centroids\": 5
  }" \
  | python -m json.tool
```

分析闭合 STL：

```bash
curl -s -X POST "$BASE_URL/data-processing/mesh/analyze" \
  -H "Content-Type: application/json" \
  -d "{
    \"path\": \"$PROJECT/test_data/physicsnemo_mesh_demo/tetrahedron_closed.stl\",
    \"device\": \"cuda:0\",
    \"include_centroids\": 4
  }" \
  | python -m json.tool
```

分析 CSV 点云，需要指定重建模式：

```bash
curl -s -X POST "$BASE_URL/data-processing/mesh/analyze" \
  -H "Content-Type: application/json" \
  -d "{
    \"path\": \"$PROJECT/test_data/physicsnemo_mesh_demo/plane_temperature_velocity.csv\",
    \"device\": \"cuda:0\",
    \"coord_cols\": \"x,y,z\",
    \"color_col\": \"temperature\",
    \"vector_cols\": \"u,v,w\",
    \"visualization_mode\": \"delaunay_2d\",
    \"projection_plane\": \"xy\"
  }" \
  | python -m json.tool
```

典型返回字段：

```json
{
  "engine": "physicsnemo.mesh",
  "num_points": 4,
  "num_cells": 2,
  "bounds": {
    "x": [0.0, 1.0],
    "y": [0.0, 1.0],
    "z": [0.0, 0.0]
  },
  "area_sum": 1.0,
  "has_point_data": ["pressure", "temperature", "u", "v", "velocity", "w"]
}
```

### 5. Mesh 物理场分析

`mesh/field-analysis` 支持：

- `temperature_gradient`
- `stress_gradient`
- `scalar_gradient`
- `surface_integral`
- `flux_integral`

温度梯度，CSV 点云先 Delaunay 成 mesh：

```bash
curl -s -X POST "$BASE_URL/data-processing/mesh/field-analysis" \
  -H "Content-Type: application/json" \
  -d "{
    \"path\": \"$PROJECT/test_data/physicsnemo_mesh_demo/plane_temperature_velocity.csv\",
    \"device\": \"cuda:0\",
    \"coord_cols\": \"x,y,z\",
    \"color_col\": \"temperature\",
    \"visualization_mode\": \"delaunay_2d\",
    \"projection_plane\": \"xy\",
    \"operation\": \"temperature_gradient\",
    \"field_key\": \"temperature\",
    \"sample_limit\": 5
  }" \
  | python -m json.tool
```

对 VTP 中的温度做表面积分：

```bash
curl -s -X POST "$BASE_URL/data-processing/mesh/field-analysis" \
  -H "Content-Type: application/json" \
  -d "{
    \"path\": \"$PROJECT/test_data/physicsnemo_mesh_demo/square_field.vtp\",
    \"device\": \"cuda:0\",
    \"operation\": \"surface_integral\",
    \"field_key\": \"temperature\"
  }" \
  | python -m json.tool
```

对 VTP 中的速度分量做通量积分：

```bash
curl -s -X POST "$BASE_URL/data-processing/mesh/field-analysis" \
  -H "Content-Type: application/json" \
  -d "{
    \"path\": \"$PROJECT/test_data/physicsnemo_mesh_demo/square_field.vtp\",
    \"device\": \"cuda:0\",
    \"operation\": \"flux_integral\",
    \"vector_keys\": [\"u\", \"v\", \"w\"]
  }" \
  | python -m json.tool
```

对 VTP 中的三分量向量场做通量积分：

```bash
curl -s -X POST "$BASE_URL/data-processing/mesh/field-analysis" \
  -H "Content-Type: application/json" \
  -d "{
    \"path\": \"$PROJECT/test_data/physicsnemo_mesh_demo/square_field.vtp\",
    \"device\": \"cuda:0\",
    \"operation\": \"flux_integral\",
    \"field_key\": \"velocity\"
  }" \
  | python -m json.tool
```

## 设计边界

PyVista 仍负责文件读取、点云重建和可视化；`physicsnemo.mesh` 负责 GPU 张量化 mesh 分析、几何统计、场梯度和积分；`physicsnemo.sym` 保留给 STL 几何采样和 PDE 几何域相关能力。

点云只有 points、没有 cells 时，不能可靠计算面积、法向、积分和梯度。因此 CSV 点云进入 `mesh/analyze` 或 `mesh/field-analysis` 时，必须指定 `visualization_mode=delaunay_2d` 或 `surface_reconstruction`。
