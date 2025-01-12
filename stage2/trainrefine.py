import argparse
from sugar_utils.general_utils import str2bool
from sugar_trainers.coarse_density import coarse_training_with_density_regularization
from sugar_trainers.coarse_sdf import coarse_training_with_sdf_regularization
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
from sugar_trainers.refine import refined_training
from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar

from datetime import datetime
import os
import yaml
from arguments import ModelParams, PipelineParams, GenerateCamParams, GuidanceParams
from arguments import OptimizationParams as OptimizationParams_arg

class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full SuGaR model.')
    
    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str, default='/',
                        help='(Required) path to the scene data to use.')  
    parser.add_argument('-c', '--checkpoint_path',
                        type=str, 
                        default='/',
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=5000, 
                        help='iteration to load.')
    
    # Regularization for coarse SuGaR
    parser.add_argument('-r', '--regularization_type', type=str,
                        default='density',
                        help='(Required) Type of regularization to use for coarse SuGaR. Can be "sdf" or "density". ' 
                        'For reconstructing detailed objects centered in the scene with 360° coverage, "density" provides a better foreground mesh. '
                        'For a stronger regularization and a better balance between foreground and background, choose "sdf".')
    
    # Extract mesh
    parser.add_argument('-l', '--surface_level', type=float, default=0.3, 
                        help='Surface level to extract the mesh at. Default is 0.3')
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=5_000_000, 
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, 
                        help='If True, center the bbox. Default is False.')
    
    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=55, 
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=5_000, 
                        help='Number of refinement iterations.')
    
    # (Optional) Parameters for textured mesh extraction
    parser.add_argument('-t', '--export_uv_textured_mesh', type=str2bool, default=True, 
                        help='If True, will export a textured mesh as an .obj file from the refined SuGaR model. '
                        'Computing a traditional colored UV texture should take less than 10 minutes.')
    parser.add_argument('--square_size',
                        default=10, type=int, help='Size of the square to use for the UV texture.')
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False, 
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                        'This step takes a few minutes and is not needed in general, as it can also be risky. '
                        'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')
    
    # (Optional) PLY file export
    parser.add_argument('--export_ply', type=str2bool, default=True,
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training. '
                        'This file can be large (+/- 500MB), but is needed for using the dedicated viewer. Default is True.')
    
    # (Optional) Default configurations
    parser.add_argument('--low_poly', type=str2bool, default=False, 
                        help='Use standard config for a low poly mesh, with 200k vertices and 6 Gaussians per triangle.')
    parser.add_argument('--high_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--refinement_time', type=str, default=None, 
                        help="Default configs for time to spend on refinement. Can be 'short', 'medium' or 'long'.")
      
    # Evaluation split
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')


    parser.add_argument('--lucidopt', type=str, default='./lucidconfigs/lowarm.yaml', help='path of config.')

    parser.add_argument('--coarse_mesh_path', type=str, default='.obj', help='path of config.')
    parser.add_argument('--prompt', type=str, default='a fox', help='prompt.')
    

    # ====================SDS function====================
    parser_lucid = argparse.ArgumentParser(description="Training script parameters")

    parser_lucid.add_argument('--opt', type=str, default=None)
    parser_lucid.add_argument('--ip', type=str, default="127.0.0.1")
    parser_lucid.add_argument('--port', type=int, default=6009)
    parser_lucid.add_argument('--debug_from', type=int, default=-1)
    parser_lucid.add_argument('--seed', type=int, default=0)
    parser_lucid.add_argument('--detect_anomaly', action='store_true', default=False)
    parser_lucid.add_argument("--test_ratio", type=int, default=5) # [2500,5000,7500,10000,12000]
    parser_lucid.add_argument("--save_ratio", type=int, default=2) # [10000,12000]
    parser_lucid.add_argument("--save_video", type=bool, default=False)
    parser_lucid.add_argument("--quiet", action="store_true")
    parser_lucid.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser_lucid.add_argument("--start_checkpoint", type=str, default=None)
    parser_lucid.add_argument("--prompt", type=str, default=None)
    parser_lucid.add_argument("--coarse_mesh_path", type=str, default=None)

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    # args_lucid, leftover_lucid = parser_lucid.parse_known_args()

    if args.low_poly:
        args.n_vertices_in_mesh = 200_000
        args.gaussians_per_triangle = 6
        print('Using low poly config.')
    if args.high_poly:
        args.n_vertices_in_mesh = 1_000_000
        args.gaussians_per_triangle = 1
        print('Using high poly config.')
    if args.refinement_time == 'short':
        args.refinement_iterations = 2_000
        print('Using short refinement time.')
    if args.refinement_time == 'medium':
        args.refinement_iterations = 7_000
        print('Using medium refinement time.')
    if args.refinement_time == 'long':
        args.refinement_iterations = 15_000
        print('Using long refinement time.')
    if args.export_uv_textured_mesh:
        print('Will export a UV-textured mesh as an .obj file.')
    if args.export_ply:
        print('Will export a ply file with the refined 3D Gaussians at the end of the training.')

    import yaml
    with open(args.lucidopt) as f:
        lucid_opts = yaml.load(f, Loader=yaml.FullLoader)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # workspace = lucid_opts['ModelParams']['workspace']

    workspace = args.prompt.replace(' ', '_')

    output_dir_all = f'./output/{workspace}@{timestamp}'
    os.makedirs(output_dir_all, exist_ok=True)
    checkpoint_path_list = args.coarse_mesh_path.split('/coarse_mesh/')
    checkpoint_path = os.path.join('/', checkpoint_path_list[0])

    
    




    # ----- Refine SuGaR -----
    output_dir_refined = os.path.join(output_dir_all, 'refine')
    os.makedirs(output_dir_refined, exist_ok=True)
    refined_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': checkpoint_path,
        'mesh_path': args.coarse_mesh_path,      
        'output_dir': output_dir_refined,
        'iteration_to_load': args.iteration_to_load,
        'normal_consistency_factor': 0.3,    
        'gaussians_per_triangle': args.gaussians_per_triangle,        
        'n_vertices_in_fg': args.n_vertices_in_mesh,
        'refinement_iterations': args.refinement_iterations,
        'bboxmin': args.bboxmin,
        'bboxmax': args.bboxmax,
        'export_ply': args.export_ply,
        'eval': args.eval,
        'gpu': args.gpu,
        'lucidopt': args.lucidopt,
        'prompt': args.prompt,
        "parser_lucid": parser_lucid
    })
    refined_sugar_path = refined_training(refined_args)
        