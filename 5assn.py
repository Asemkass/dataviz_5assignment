import open3d as o3d
import numpy as np
import copy

def print_info(geometry, step_name):
    print(f"\n=== {step_name} ===")
    
    if hasattr(geometry, 'vertices'):
        print(f"Number of vertices: {len(geometry.vertices)}")
    elif hasattr(geometry, 'points'):
        print(f"Number of points: {len(geometry.points)}")
    
    if hasattr(geometry, 'triangles'):
        print(f"Number of triangles: {len(geometry.triangles)}")
    
    if str(type(geometry)).find('VoxelGrid') != -1:
        try:
            voxels = geometry.get_voxels()
            print(f"Number of voxels: {len(voxels)}")
        except:
            print("Number of voxels: Available (cannot access count)")
    
    if hasattr(geometry, 'vertex_colors') and len(geometry.vertex_colors) > 0:
        print("Has vertex colors: Yes")
    elif hasattr(geometry, 'colors') and len(geometry.colors) > 0:
        print("Has colors: Yes")
    else:
        print("Has colors: No")
    
    if hasattr(geometry, 'vertex_normals') and len(geometry.vertex_normals) > 0:
        print("Has normals: Yes")
    elif hasattr(geometry, 'normals') and len(geometry.normals) > 0:
        print("Has normals: Yes")
    else:
        print("Has normals: No")

def main():
    print("TASK 1: LOADING AND VISUALIZATION")
    
    mesh = o3d.io.read_triangle_mesh("Gorilla.stl")
    
    if len(mesh.vertices) == 0:
        print("Failed to load mesh, creating sample mesh...")
       
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        mesh.translate([-0.5, -0.5, -0.5])  
    
    print_info(mesh, "Original Model")
    
    print("Displaying original model...")
    o3d.visualization.draw_geometries([mesh], window_name="Task 1: Original Model")
    
    print("TASK 2: CONVERSION TO POINT CLOUD")
    
    print("Sampling points from mesh...")
    point_cloud = mesh.sample_points_uniformly(number_of_points=15000)
    
    print_info(point_cloud, "Point Cloud")
    
    print("Displaying point cloud...")
    o3d.visualization.draw_geometries([point_cloud], window_name="Task 2: Point Cloud")
    
    print("TASK 3: SURFACE RECONSTRUCTION FROM POINT CLOUD")
    
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    point_cloud.orient_normals_consistent_tangent_plane(k=30)
    
    print("Performing Poisson surface reconstruction...")
    mesh_reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=6, width=0, scale=1.1, linear_fit=False)
    
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.2)  
    vertices_to_remove = densities < density_threshold
    mesh_reconstructed.remove_vertices_by_mask(vertices_to_remove)
    
    mesh_reconstructed.remove_degenerate_triangles()
    mesh_reconstructed.remove_duplicated_triangles()
    mesh_reconstructed.remove_duplicated_vertices()
    mesh_reconstructed.remove_non_manifold_edges()
    
    print_info(mesh_reconstructed, "Reconstructed Mesh")
    
    print("Displaying reconstructed mesh...")
    o3d.visualization.draw_geometries([mesh_reconstructed], window_name="Task 3: Reconstructed Mesh")
    
    print("TASK 4: VOXELIZATION")
    
    voxel_size = 0.5
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)
    
    print_info(voxel_grid, f"Voxel Grid (size={voxel_size})")
    
    print("Displaying voxel grid...")
    o3d.visualization.draw_geometries([voxel_grid], window_name="Task 4: Voxel Grid")
    
    print("TASK 5: ADDING A PLANE")
    
    points = np.asarray(point_cloud.points)
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    
    padding = 0.2
    height = (y_max - y_min) + 2 * padding
    depth = (z_max - z_min) + 2 * padding
    
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=0.02, height=height, depth=depth)
    
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2
    plane_mesh.translate([-0.01, center_y - height/2, center_z - depth/2])
    
    plane_mesh.paint_uniform_color([0.6, 0.6, 0.6])  
    
    print("Created vertical clipping plane at x=0")
    print(f"Plane vertices: {len(plane_mesh.vertices)}")
    print(f"Plane triangles: {len(plane_mesh.triangles)}")
    print(f"Plane dimensions: {0.02:.3f} x {height:.3f} x {depth:.3f}")
    
    print("Displaying object with clipping plane...")
    o3d.visualization.draw_geometries([point_cloud, plane_mesh], window_name="Task 5: Object with Clipping Plane")
    
    print("TASK 6: SURFACE CLIPPING")
    
    points = np.asarray(point_cloud.points)
    
    mask = points[:, 0] <= 0
    clipped_points = points[mask]
    
    clipped_cloud = o3d.geometry.PointCloud()
    clipped_cloud.points = o3d.utility.Vector3dVector(clipped_points)
    
    if len(point_cloud.colors) > 0:
        colors = np.asarray(point_cloud.colors)
        clipped_colors = colors[mask]
        clipped_cloud.colors = o3d.utility.Vector3dVector(clipped_colors)
    
    if len(point_cloud.normals) > 0:
        normals = np.asarray(point_cloud.normals)
        clipped_normals = normals[mask]
        clipped_cloud.normals = o3d.utility.Vector3dVector(clipped_normals)
    
    print_info(clipped_cloud, "Clipped Point Cloud")
    print(f"Original points: {len(points)}")
    print(f"Remaining points after clipping: {len(clipped_points)}")
    
    print("Displaying clipped model...")
    o3d.visualization.draw_geometries([clipped_cloud], window_name="Task 6: Clipped Model")
    
    print("TASK 7: WORKING WITH COLOR AND EXTREMES")
    
    points = np.asarray(point_cloud.points)
    
    z_values = points[:, 2]
    z_min, z_max = np.min(z_values), np.max(z_values)
    
    normalized_z = (z_values - z_min) / (z_max - z_min) if z_max != z_min else np.zeros_like(z_values)
    
    colors = np.zeros((len(points), 3))
    colors[:, 0] = normalized_z       
    colors[:, 2] = 1.0 - normalized_z  
    
    colored_cloud = copy.deepcopy(point_cloud)
    colored_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    z_min_idx = np.argmin(z_values)
    z_max_idx = np.argmax(z_values)
    
    min_point = points[z_min_idx]
    max_point = points[z_max_idx]
    
    print(f"Z-axis extremes:")
    print(f"Minimum Z point: {min_point} (Z = {z_min:.3f})")
    print(f"Maximum Z point: {max_point} (Z = {z_max:.3f})")
    
    min_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    min_sphere.translate(min_point)
    min_sphere.paint_uniform_color([0, 1, 0])  
    
    max_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    max_sphere.translate(max_point)
    max_sphere.paint_uniform_color([1, 0, 1]) 
    
    print("Displaying colored point cloud with extreme points marked...")
    o3d.visualization.draw_geometries([colored_cloud, min_sphere, max_sphere], 
                                    window_name="Task 7: Colored Cloud with Extremes")
    
    min_cube = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
    min_cube.translate(min_point - [0.05, 0.05, 0.05])
    min_cube.paint_uniform_color([1, 1, 0]) 
    
    max_cube = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
    max_cube.translate(max_point - [0.05, 0.05, 0.05])
    max_cube.paint_uniform_color([0, 1, 1])  
    
    min_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(min_cube)
    max_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(max_cube)
    
    print("Displaying colored point cloud with wireframe extreme markers...")
    o3d.visualization.draw_geometries([colored_cloud, min_wireframe, max_wireframe], 
                                    window_name="Task 7: Colored Cloud with Wireframe Extremes")
    
    
if __name__ == "__main__":
    main()