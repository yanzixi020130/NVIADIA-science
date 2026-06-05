# NVIDIA Science Data Processing

本项目是一个面向科学数据处理与 3D 几何可视化的 FastAPI 服务，主要能力包括：

- 文件上传与文件列表查询。
- CSV/TXT/DAT 表格字段识别、点云预览、时间序列点云动画。
- 点云重建为三角网格，并输出 PNG 预览图与 VTP 网格文件。
- STL/OBJ/PLY/VTP/VTK 等几何文件读取、预览与分析。
- 基于 `physicsnemo.mesh` 的 mesh 几何统计、质量检查和物理场分析。
- 基于大模型的自然语言生成 3D 点云或三角网格，接口为 `/data-processing/text-to-3d`。

当前服务默认端口为 `1101`，核心接口集中在 `/data-processing` 路由下。

## 目录结构

```text
.
├── main.py                         # FastAPI 主入口
├── src/
│   ├── data_processing_module.py   # 数据处理、点云/mesh、text-to-3d 接口
│   ├── point_cloud_visualizer.py   # PyVista 可视化与文件输出
│   ├── physicsnemo_mesh_adapter.py # PyVista 与 PhysicsNeMo Mesh 适配
│   ├── text_to_3d_generator.py     # 大模型 text-to-3d 调用、解析、校验与参数化几何生成
│   └── data_field_mapper.py        # 表格字段自动识别
├── upload/                         # 上传和生成的数据文件目录，运行时生成
├── images/                         # PNG/GIF/VTP 可视化输出目录，运行时生成
└── test_data/                      # 示例测试数据
```

## 环境准备

推荐 Python 3.10 或 3.11。

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

如需使用 GPU 与 PhysicsNeMo Mesh，请确认 PyTorch/CUDA 可用：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 大模型配置

`text-to-3d` 使用 OpenAI-compatible Chat Completions 接口。复制 `.env.example` 为 `.env` 后配置：

```env
TEXT_TO_3D_LLM_BASE_URL=http://www.science42.vip:40200/v1
TEXT_TO_3D_LLM_API_KEY=replace-with-your-api-key
TEXT_TO_3D_LLM_MODEL=SE_V0.0
```

说明：

- `TEXT_TO_3D_LLM_BASE_URL`：大模型服务地址，可选；默认 `http://www.science42.vip:40200/v1`。
- `TEXT_TO_3D_LLM_API_KEY`：大模型 API Key，必填。
- `TEXT_TO_3D_LLM_MODEL`：模型名称，可选；默认 `SE_V0.0`。

`.env` 已在 `.gitignore` 中，避免密钥提交到仓库。

## 启动服务

```bash
python main.py
```

默认访问地址：

```text
http://localhost:1101
```

也可以通过环境变量修改端口：

```bash
PORT=1102 python main.py
```

Windows PowerShell：

```powershell
$env:PORT="1102"
python main.py
```

## 输出文件位置

服务会按 `taskid` 保存文件：

```text
upload/{taskid}/data_processing/                 # 上传文件、生成的 JSON/VTP 数据
images/data_processing/{taskid}/                 # 生成的 PNG/GIF/VTP 可视化文件
```

例如 `taskid=demo` 的 text-to-3d 图片通常位于：

```text
images/data_processing/demo/gen_<timestamp>_<id>_triangle_mesh.png
```

接口返回中也会包含：

- `image_path`：服务器本地图片路径。
- `image_url`：可通过服务访问的图片 URL，例如 `/images/data_processing/demo/xxx.png`。
- `path`：生成或上传的数据文件路径。
- `mesh_url`：可通过服务访问的 VTP 网格文件 URL。

## 参数说明约定

每个接口参数按三类说明：

- 必填：请求必须提供。
- 可选：请求可以提供；未提供时使用默认值。
- 不可填：客户端不应传入；由服务生成或只出现在响应中。

## 通用接口

### GET `/roles`

获取当前团队角色配置。

参数：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| 无 | - | - | 无请求参数 |

调用示例：

```bash
curl -s http://localhost:1101/roles
```

### WebSocket `/start`

启动原有团队分析流程。

首条消息 JSON 参数：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `idea` | string | 必填 | 用户任务描述 |
| `taskid` | string | 必填 | 任务 ID，用于关联上传文件 |
| `user_name` | string | 必填 | 用户名 |
| `file_metadata` | array | 必填 | 文件元数据列表 |

服务端会通过 WebSocket 发送 `[start]`、`[end]`、`[Pending]` 等状态消息。

不可填：

- `n_round`：由服务端根据角色数量计算。
- `team`：由服务端创建。

### POST `/uploadFile`

上传普通文件到 `upload/{taskid}/`。

请求类型：`multipart/form-data`

参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `files` | file[] | 必填 | - | 一个或多个上传文件 |
| `taskid` | string | 必填 | - | 保存目录名 |

不可填：

- `path`：由服务端根据 `taskid` 和文件名生成。

调用示例：

```bash
curl -s -X POST http://localhost:1101/uploadFile \
  -F "taskid=demo" \
  -F "files=@test_data/physicsnemo_mesh_demo/tetrahedron_closed.stl"
```

### POST `/files`

列出 `upload/{taskid}/` 下的普通上传文件。

请求类型：`multipart/form-data`

参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `taskid` | string | 必填 | - | 任务 ID |

调用示例：

```bash
curl -s -X POST http://localhost:1101/files \
  -F "taskid=demo"
```

## Data Processing 接口

### GET `/data-processing/health`

数据处理模块健康检查。

参数：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| 无 | - | - | 无请求参数 |

调用示例：

```bash
curl -s http://localhost:1101/data-processing/health
```

### GET `/data-processing/physicsnemo/status`

检查 `physicsnemo`、`physicsnemo.sym`、`physicsnemo.mesh` 和本地 tessellation 是否可用。

参数：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| 无 | - | - | 无请求参数 |

调用示例：

```bash
curl -s http://localhost:1101/data-processing/physicsnemo/status
```

### POST `/data-processing/inspect-table`

上传 CSV/TXT/DAT 表格并检查字段映射，不生成图片。

请求类型：`multipart/form-data`

参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `taskid` | string | 必填 | - | 任务 ID |
| `file` | file | 必填 | - | CSV/TXT/DAT 文件 |
| `coord_cols` | string | 可选 | 自动识别 | 坐标列，格式如 `x,y,z` |
| `color_col` | string | 可选 | 自动识别 | 标量着色列，如 `temperature` |
| `time_col` | string | 可选 | 自动识别 | 时间列，用于动画 |
| `vector_cols` | string | 可选 | 自动识别 | 向量列，格式如 `u,v,w` |

不可填：

- `dataset_id`：服务端生成。
- `path`：服务端保存文件后生成。

调用示例：

```bash
curl -s -X POST http://localhost:1101/data-processing/inspect-table \
  -F "taskid=mesh-demo" \
  -F "file=@test_data/physicsnemo_mesh_demo/plane_temperature_velocity.csv" \
  -F "coord_cols=x,y,z" \
  -F "color_col=temperature" \
  -F "vector_cols=u,v,w"
```

### POST `/data-processing/upload-preview`

上传数据文件并生成预览。CSV/TXT/DAT 点云可生成静态 PNG、可选 GIF 动画、可选重建 mesh 与 mesh 分析。STL/OBJ/PLY/VTP/VTK/JSON 文件会生成基础预览信息。

请求类型：`multipart/form-data`

参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `file` | file | 必填 | - | 上传文件 |
| `taskid` | string | 可选 | `default` | 任务 ID |
| `coord_cols` | string | 可选 | 自动识别 | 点云坐标列，如 `x,y,z` |
| `color_col` | string | 可选 | 自动识别 | 标量着色列 |
| `time_col` | string | 可选 | 自动识别 | 时间列，有值时生成 GIF |
| `vector_cols` | string | 可选 | 自动识别 | 向量列，如 `u,v,w` |
| `max_points` | int | 可选 | `5000` | 最大预览点数 |
| `max_frames` | int | 可选 | `60` | 最大动画帧数 |
| `visualization_mode` | string | 可选 | `point_cloud` | `point_cloud`、`delaunay_2d`、`surface_reconstruction` |
| `projection_plane` | string | 可选 | `xy` | `xy`、`xz`、`yz`、`pca`，用于 `delaunay_2d` |
| `delaunay_alpha` | float | 可选 | `null` | 2D Delaunay alpha 参数 |
| `surface_method` | string | 可选 | `reconstruct_surface` | `reconstruct_surface` 或 `delaunay_3d` |
| `surface_alpha` | float | 可选 | `null` | 3D Delaunay alpha 参数 |
| `nbr_sz` | int | 可选 | `20` | 表面重建邻域大小，范围 `1..200` |
| `sample_spacing` | float | 可选 | `null` | 表面重建采样间距 |
| `show_edges` | bool | 可选 | `false` | 是否显示网格边 |
| `show_grid` | bool | 可选 | `true` | 是否显示坐标网格 |
| `colormap` | string | 可选 | `viridis` | 标量着色使用的 colormap，例如 `inferno`、`plasma`、`magma`、`turbo` |
| `camera_zoom` | float | 可选 | `1.0` | 视角缩放比例；`0.5` 表示缩小一半，`1.5` 表示放大 |
| `canvas_size` | int | 可选 | `1000` | 输出画布边长（像素），始终生成正方形画布，范围 `200..4000` |
| `plot_title` | string | 可选 | 自动生成 | PNG/GIF 图像标题 |
| `x_axis_title` | string | 可选 | PyVista 默认值 | X 轴标题 |
| `y_axis_title` | string | 可选 | PyVista 默认值 | Y 轴标题 |
| `z_axis_title` | string | 可选 | PyVista 默认值 | Z 轴标题 |
| `mesh_opacity` | float | 可选 | `1.0` | mesh 透明度，范围 `0..1` |
| `mesh_color` | string | 可选 | `#9bbfc2` | 无标量场时的 mesh 颜色 |
| `mesh_edge_color` | string | 可选 | `#111827` | mesh 边颜色 |
| `mesh_edge_width` | float | 可选 | `1.0` | mesh 边宽，范围 `0.1..10` |
| `include_mesh_analysis` | bool | 可选 | `true` | 是否尝试 PhysicsNeMo mesh 分析 |
| `mesh_analysis_device` | string | 可选 | `cuda` | `cuda`、`cuda:0` 或 `cpu` |
| `mesh_analysis_centroids` | int | 可选 | `5` | 返回的 cell centroid 样本数，范围 `0..100` |

不可填：

- `dataset_id`、`image_path`、`image_url`、`mesh_path`、`mesh_url`：由服务端生成并返回。

点云预览示例：

```bash
curl -s -X POST http://localhost:1101/data-processing/upload-preview \
  -F "taskid=mesh-demo" \
  -F "file=@test_data/physicsnemo_mesh_demo/plane_temperature_velocity.csv" \
  -F "coord_cols=x,y,z" \
  -F "color_col=temperature" \
  -F "vector_cols=u,v,w" \
  -F "visualization_mode=point_cloud" \
  -F "show_grid=true" \
  -F "colormap=inferno" \
  -F "camera_zoom=0.5" \
  -F "canvas_size=800" \
  -F "plot_title=Temperature Field" \
  -F "x_axis_title=X / m" \
  -F "y_axis_title=Y / m" \
  -F "z_axis_title=Z / m"
```

点云重建为三角网格示例：

```bash
curl -s -X POST http://localhost:1101/data-processing/upload-preview \
  -F "taskid=mesh-demo" \
  -F "file=@test_data/physicsnemo_mesh_demo/plane_temperature_velocity.csv" \
  -F "coord_cols=x,y,z" \
  -F "color_col=temperature" \
  -F "visualization_mode=delaunay_2d" \
  -F "projection_plane=xy" \
  -F "show_edges=true" \
  -F "include_mesh_analysis=false"
```

### POST `/data-processing/text-to-3d`

调用大模型，将自然语言转换为 3D 点云或三角网格。服务端会强制要求模型返回 JSON，再进行几何校验、PyVista 转换、PNG 渲染和 VTP 保存。

请求类型：`application/json`

参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `prompt` | string | 必填 | - | 自然语言生成需求。建议包含对象、尺寸、分辨率、物理场、输出类型 |
| `taskid` | string | 可选 | `default` | 任务 ID |
| `output_type` | string | 可选 | `auto` | `auto`、`point_cloud`、`triangle_mesh`、`parametric_surface` |
| `max_points` | int | 可选 | `3000` | 允许的最大点数，范围 `1..200000` |
| `max_faces` | int | 可选 | `6000` | 允许的最大三角面数，范围 `1..200000` |
| `include_mesh_analysis` | bool | 可选 | `true` | 是否对三角网格执行 PhysicsNeMo mesh 分析 |
| `device` | string | 可选 | `cuda` | mesh 分析设备，常用 `cuda`、`cuda:0`、`cpu` |
| `temperature` | float | 可选 | `0.2` | 大模型采样温度，范围 `0..2` |
| `timeout` | float | 可选 | `60.0` | 大模型请求超时秒数，范围 `1..300` |
| `show_edges` | bool | 可选 | `true` | 渲染时是否显示三角网格边 |
| `show_grid` | bool | 可选 | `true` | 渲染时是否显示坐标网格 |
| `colormap` | string | 可选 | `viridis` | 标量着色使用的 colormap，例如 `inferno`、`plasma`、`magma`、`turbo` |
| `camera_zoom` | float | 可选 | `1.0` | 视角缩放比例；`0.5` 表示缩小一半，`1.5` 表示放大 |
| `canvas_size` | int | 可选 | `1000` | 输出画布边长（像素），始终生成正方形画布，范围 `200..4000` |
| `plot_title` | string | 可选 | 模型描述或几何类型 | PNG 图像标题 |
| `x_axis_title` | string | 可选 | PyVista 默认值 | X 轴标题 |
| `y_axis_title` | string | 可选 | PyVista 默认值 | Y 轴标题 |
| `z_axis_title` | string | 可选 | PyVista 默认值 | Z 轴标题 |
| `mesh_opacity` | float | 可选 | `1.0` | mesh 透明度，范围 `0..1` |
| `mesh_color` | string | 可选 | `#9bbfc2` | 无标量场时的 mesh 颜色 |
| `mesh_edge_color` | string | 可选 | `#111827` | mesh 边颜色 |
| `mesh_edge_width` | float | 可选 | `1.0` | mesh 边宽，范围 `0.1..10` |
| `include_raw_response` | bool | 可选 | `false` | 是否返回大模型原始输出，调试时使用 |

不可填：

- `dataset_id`：服务端生成。
- `generated_json_path`：服务端生成。
- `path`：服务端生成的 VTP 文件路径。
- `image_path`、`image_url`：服务端渲染后生成。
- `mesh_path`、`mesh_url`：服务端保存后生成。
- `mesh_analysis`：服务端分析后生成。

模型允许输出的 JSON 类型：

- `point_cloud`：直接给 `points` 和可选 `scalars`。
- `triangle_mesh`：给 `points`、`faces` 和可选 `scalars`。
- `parametric_surface`：推荐复杂几何使用。模型只给形状、参数、分辨率，后端确定性生成点和三角面。

推荐提示词结构：

```text
生成对象：具体几何对象。
几何参数：尺寸、半径、展长、弦长、弯度、波幅等。
网格参数：采样点数或 resolution。
物理场：temperature、pressure、velocity 等字段及变化规律。
输出要求：返回 parametric_surface / triangle_mesh / point_cloud；只返回合法 JSON，不要解释文字。
```

机翼三角网格示例，推荐用于本项目：

```bash
curl -s -X POST http://localhost:1101/data-processing/text-to-3d \
  -H "Content-Type: application/json" \
  -d "{\"taskid\":\"demo\",\"prompt\":\"生成一个三维机翼上表面的三角网格，用于科学可视化和后续 mesh 分析。机翼展长 5.0 米，根弦长 1.4 米，梢弦长 0.65 米，最大弯度 0.08，翼尖扭转角 5 度。网格分辨率使用展向 72 个采样点、弦向 28 个采样点。请优先输出 parametric_surface 类型，shape 使用 wing，parameters 中包含 span、root_chord、tip_chord、camber、twist，resolution 为 [72,28]。包含 temperature 标量场，温度沿 x 方向从 280K 线性升高到 360K。只返回合法 JSON，不要输出解释文字。\",\"output_type\":\"auto\",\"max_points\":3000,\"max_faces\":6000,\"include_mesh_analysis\":false,\"show_edges\":true,\"show_grid\":true,\"colormap\":\"inferno\",\"camera_zoom\":0.5,\"canvas_size\":800,\"plot_title\":\"Wing Temperature Field\",\"x_axis_title\":\"X / m\",\"y_axis_title\":\"Y / m\",\"z_axis_title\":\"Z / m\"}"
```

机翼点云示例：

```bash
curl -s -X POST http://localhost:1101/data-processing/text-to-3d \
  -H "Content-Type: application/json" \
  -d "{\"taskid\":\"demo\",\"prompt\":\"生成一个机翼表面的三维点云，用于科学可视化。机翼展长 5.0 米，根弦长 1.4 米，梢弦长 0.65 米，最大弯度 0.08，翼尖扭转角 5 度。采样点数量约 1500 个。每个点包含 x、y、z 坐标，并包含 temperature 标量场，温度沿 x 方向从 280K 到 360K 线性变化。输出 point_cloud 类型，只返回合法 JSON，不要解释文字。\",\"output_type\":\"point_cloud\",\"max_points\":3000,\"include_mesh_analysis\":false}"
```

球面三角网格示例：

```bash
curl -s -X POST http://localhost:1101/data-processing/text-to-3d \
  -H "Content-Type: application/json" \
  -d "{\"taskid\":\"sphere-demo\",\"prompt\":\"生成一个半径 1.0 米的低分辨率球面三角网格，用于验证 3D mesh 渲染。请输出 parametric_surface 类型，shape 使用 sphere，resolution 为 [48,24]，parameters 包含 radius=1.0。包含 temperature 标量场，温度沿 z 方向线性变化。只返回合法 JSON。\",\"output_type\":\"auto\",\"max_points\":1500,\"max_faces\":3000,\"include_mesh_analysis\":false}"
```

典型响应字段：

```json
{
  "success": true,
  "taskid": "demo",
  "dataset_id": "gen_...",
  "generated_type": "triangle_mesh",
  "num_points": 2016,
  "num_faces": 3834,
  "scalar_fields": ["temperature"],
  "generated_json_path": ".../upload/demo/data_processing/gen_..._triangle_mesh.json",
  "path": ".../upload/demo/data_processing/gen_..._triangle_mesh.vtp",
  "image_url": "/images/data_processing/demo/gen_..._triangle_mesh.png",
  "mesh_url": "/images/data_processing/demo/gen_..._triangle_mesh.vtp"
}
```

### POST `/data-processing/geometry/sample`

对 STL 几何进行边界点和内部点采样。优先使用 `physicsnemo.sym.geometry.tessellation_warp.Tessellation`，不可用时回退到本地 `src.geometry.Tessellation`。

请求类型：`application/json`

参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `taskid` | string | 可选 | `default` | 任务 ID |
| `dataset_id` | string | 可选 | `null` | 已上传文件的 dataset ID 前缀 |
| `filename` | string | 可选 | `null` | 已上传文件名 |
| `path` | string | 可选 | `null` | 工作区内 STL 文件绝对或相对路径 |
| `sample_points` | int | 可选 | `10000` | 采样点数，范围 `1..200000` |
| `return_points` | int | 可选 | `5000` | 响应中返回的点数上限，范围 `1..20000` |
| `include_boundary` | bool | 可选 | `true` | 是否采样边界 |
| `include_interior` | bool | 可选 | `true` | 是否采样内部 |
| `compute_sdf_derivatives` | bool | 可选 | `true` | 内部采样时是否计算 SDF 导数 |
| `device` | string | 可选 | `null` | 设备，如 `cuda:0`、`cpu` |
| `seed` | int | 可选 | `null` | 随机种子 |

路径规则：

- `path`、`dataset_id`、`filename` 至少应能定位到一个 STL 文件。
- 如果传 `path`，必须位于当前项目工作区内。

调用示例：

```bash
curl -s -X POST http://localhost:1101/data-processing/geometry/sample \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"test_data/physicsnemo_mesh_demo/tetrahedron_closed.stl\",\"sample_points\":200,\"return_points\":10,\"include_boundary\":true,\"include_interior\":true,\"compute_sdf_derivatives\":true,\"device\":\"cpu\"}"
```

### POST `/data-processing/mesh/analyze`

分析 mesh 几何统计和质量信息。支持 `.vtp`、`.vtk`、`.stl`、`.ply`、`.obj`，也支持 CSV/TXT/DAT 点云先重建为三角网格后分析。

请求类型：`application/json`

参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `taskid` | string | 可选 | `default` | 任务 ID |
| `dataset_id` | string | 可选 | `null` | 已上传或已生成文件的 ID 前缀 |
| `filename` | string | 可选 | `null` | 文件名 |
| `path` | string | 可选 | `null` | 工作区内文件路径 |
| `device` | string | 可选 | `cuda` | PhysicsNeMo mesh 设备 |
| `include_centroids` | int | 可选 | `5` | centroid 样本数，范围 `0..100` |
| `max_points` | int | 可选 | `5000` | CSV 点云最大读取点数 |
| `coord_cols` | string | 可选 | 自动识别 | CSV 坐标列，如 `x,y,z` |
| `color_col` | string | 可选 | 自动识别 | CSV 标量列 |
| `time_col` | string | 可选 | 自动识别 | CSV 时间列 |
| `vector_cols` | string | 可选 | 自动识别 | CSV 向量列 |
| `visualization_mode` | string | 可选 | `point_cloud` | CSV 点云分析 mesh 时必须用 `delaunay_2d` 或 `surface_reconstruction` |
| `projection_plane` | string | 可选 | `xy` | Delaunay 投影平面 |
| `delaunay_alpha` | float | 可选 | `null` | Delaunay alpha |
| `surface_method` | string | 可选 | `reconstruct_surface` | 表面重建方法 |
| `surface_alpha` | float | 可选 | `null` | 3D Delaunay alpha |
| `nbr_sz` | int | 可选 | `20` | 表面重建邻域大小 |
| `sample_spacing` | float | 可选 | `null` | 表面重建采样间距 |

路径规则：

- `path`、`dataset_id`、`filename` 至少应能定位到一个文件。
- 如果传 `path`，必须位于当前项目工作区内。

调用示例，分析 VTP：

```bash
curl -s -X POST http://localhost:1101/data-processing/mesh/analyze \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"test_data/physicsnemo_mesh_demo/square_field.vtp\",\"device\":\"cpu\",\"include_centroids\":5}"
```

调用示例，分析 CSV 点云重建 mesh：

```bash
curl -s -X POST http://localhost:1101/data-processing/mesh/analyze \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"test_data/physicsnemo_mesh_demo/plane_temperature_velocity.csv\",\"device\":\"cpu\",\"coord_cols\":\"x,y,z\",\"color_col\":\"temperature\",\"vector_cols\":\"u,v,w\",\"visualization_mode\":\"delaunay_2d\",\"projection_plane\":\"xy\"}"
```

### POST `/data-processing/mesh/field-analysis`

对 mesh 中的物理场执行梯度、表面积分或通量积分。

请求类型：`application/json`

该接口继承 `/mesh/analyze` 的所有定位、CSV 映射和重建参数，并额外支持：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `operation` | string | 可选 | `temperature_gradient` | `temperature_gradient`、`stress_gradient`、`scalar_gradient`、`surface_integral`、`flux_integral` |
| `field_key` | string | 可选 | `null` | 标量场或向量场字段名 |
| `vector_keys` | string[] | 可选 | `null` | 三个向量分量字段，如 `["u","v","w"]` |
| `method` | string | 可选 | `lsq` | 梯度计算方法 |
| `sample_limit` | int | 可选 | `5` | 梯度样本数，范围 `0..100` |

使用规则：

- `surface_integral` 必须提供 `field_key`。
- `flux_integral` 必须提供 `field_key` 或 `vector_keys`。
- `temperature_gradient` 默认使用 `field_key=temperature`，也可以显式传入。
- `stress_gradient` 默认使用 `field_key=stress`，也可以显式传入。

温度梯度示例：

```bash
curl -s -X POST http://localhost:1101/data-processing/mesh/field-analysis \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"test_data/physicsnemo_mesh_demo/plane_temperature_velocity.csv\",\"device\":\"cpu\",\"coord_cols\":\"x,y,z\",\"color_col\":\"temperature\",\"visualization_mode\":\"delaunay_2d\",\"projection_plane\":\"xy\",\"operation\":\"temperature_gradient\",\"field_key\":\"temperature\",\"sample_limit\":5}"
```

表面积分示例：

```bash
curl -s -X POST http://localhost:1101/data-processing/mesh/field-analysis \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"test_data/physicsnemo_mesh_demo/square_field.vtp\",\"device\":\"cpu\",\"operation\":\"surface_integral\",\"field_key\":\"temperature\"}"
```

通量积分示例：

```bash
curl -s -X POST http://localhost:1101/data-processing/mesh/field-analysis \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"test_data/physicsnemo_mesh_demo/square_field.vtp\",\"device\":\"cpu\",\"operation\":\"flux_integral\",\"vector_keys\":[\"u\",\"v\",\"w\"]}"
```

### GET `/data-processing/files`

列出 `upload/{taskid}/data_processing/` 下的数据处理文件。

查询参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `taskid` | string | 可选 | `default` | 任务 ID |

调用示例：

```bash
curl -s "http://localhost:1101/data-processing/files?taskid=demo"
```

### GET `/data-processing/demo`

返回数据处理模块的内置 HTML 演示页。

参数：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| 无 | - | - | 无请求参数 |

调用示例：

```bash
curl -s http://localhost:1101/data-processing/demo
```

## 支持的数据格式

表格与点云：

- `.csv`
- `.txt`
- `.dat`

Mesh 与几何：

- `.stl`
- `.obj`
- `.ply`
- `.vtp`
- `.vtk`

JSON 预览：

- `.json`
- `.jsonl`

## 设计边界

- PyVista 负责文件读取、点云重建、静态图和动画渲染。
- `physicsnemo.mesh` 负责 GPU/CPU mesh 张量化、几何统计、质量检查、场梯度和积分。
- `physicsnemo.sym` 或本地 `src.geometry.Tessellation` 负责 STL 几何采样。
- 点云本身没有 cells，不能可靠计算面积、法向、积分和梯度。因此 CSV 点云进入 `/mesh/analyze` 或 `/mesh/field-analysis` 时，需要使用 `visualization_mode=delaunay_2d` 或 `visualization_mode=surface_reconstruction` 先重建三角网格。
- `text-to-3d` 不直接信任大模型输出。所有点、三角面和标量场都会在服务端校验；复杂对象推荐让模型输出 `parametric_surface`，由后端确定性生成网格。
