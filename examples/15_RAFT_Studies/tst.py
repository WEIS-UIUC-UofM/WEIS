    #--------------------------------- Max Pltfrm Pitch-----------------------------------------
    
    # Normalize the data
    norm = Normalize(vmin=Max_PtfmPitch_vc_mesh.min(), vmax=Max_PtfmPitch_vc_mesh.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(Max_PtfmPitch_vc_mesh, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, Max_PtfmPitch_vc_mesh.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, Max_PtfmPitch_vc_mesh.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_title('$\mathrm{Max\,\,Ptfm\,\,Pitch\,\,[deg]}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(Max_PtfmPitch_vc_mesh.shape[0]):
        for j in range(Max_PtfmPitch_vc_mesh.shape[1]):
            ax.text(j, i, f'{Max_PtfmPitch_vc_mesh[i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'Max_PtfmPitch_vc_mesh_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'Max_PtfmPitch_vc_mesh_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.55 )  

    # Assuming x and F are your arrays
    delta_Y_pntoon_upper_vec_mesh = np.gradient(Y_pntoon_upper_vec_mesh, axis=1)  # Compute the gradient of x along the second axis
    delta_Max_PtfmPitch_vc_mesh_over_up_Pntoon= np.gradient(Max_PtfmPitch_vc_mesh, axis=1) / delta_Y_pntoon_upper_vec_mesh  # Compute the derivative of F with respect to x
    delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize=delta_Max_PtfmPitch_vc_mesh_over_mainclmn*10.0/6.8 #D_baseline=10, thet_baseline=6.8

    # Normalize the data
    norm = Normalize(vmin=delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize.min(), vmax=-delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize.min())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$', fontsize=18)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$', fontsize=18)
    #ax.set_title('$\frac{\Delta \theta_p/\theta_p^0}{\Delta D_{\mathrm{Main}}/D_{\mathrm{Main}}^0}$', fontsize=15)
    ax.set_title(r'$\frac{{\Delta \theta_p^{\mathrm{max}}/\theta_p^0}}{{\Delta D_{\mathrm{Main}}/D_{\mathrm{Main}}^0}}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=35, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize.shape[0]):
        for j in range(delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize.shape[1]):
            ax.text(j, i, f'{delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize[i, j]:.2f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_Max_PtfmPitch_vc_mesh_over_mainclmn_normalize_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  

    # Assuming x and F are your arrays
    delta_Y_pntoon_lower_vec_mesh = np.gradient(Y_pntoon_lower_vec_mesh, axis=0)  # Compute the gradient of x along the second axis
    delta_Max_PtfmPitch_vc_mesh_over_low_ontoon = np.gradient(Max_PtfmPitch_vc_mesh, axis=0) / delta_Y_pntoon_lower_vec_mesh  # Compute the derivative of F with respect to x
    delta_Max_PtfmPitch_vc_mesh_over_low_ontoon_normalize=delta_Max_PtfmPitch_vc_mesh_over_low_ontoon*12.5/6.8 #D_baseline=10, thet_baseline=6.8

    # Normalize the data
    norm = Normalize(vmin=delta_Max_PtfmPitch_vc_mesh_over_low_ontoon_normalize.min(), vmax=-delta_Max_PtfmPitch_vc_mesh_over_low_ontoon_normalize.min())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(delta_Max_PtfmPitch_vc_mesh_over_low_ontoon_normalize, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, delta_Max_PtfmPitch_vc_mesh_over_low_ontoon_normalize.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, delta_Max_PtfmPitch_vc_mesh_over_low_ontoon_normalize.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=18)
    #ax.set_title('$\frac{\Delta \theta_p/\theta_p^0}{\Delta D_{\mathrm{Main}}/D_{\mathrm{Main}}^0}$', fontsize=15)
    ax.set_title(r'$\frac{\Delta \theta_p^{\mathrm{max}}/\theta_p^0}{\Delta D_{\mathrm{Side}}/D_{\mathrm{Side}}^0}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=35, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(delta_Max_PtfmPitch_vc_mesh_over_low_ontoon_normalize.shape[0]):
        for j in range(delta_Max_PtfmPitch_vc_mesh_over_low_ontoon_normalize.shape[1]):
            ax.text(j, i, f'{delta_Max_PtfmPitch_vc_mesh_over_low_ontoon_normalize[i, j]:.2f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_Max_PtfmPitch_vc_mesh_over_Sideclmn_normalize_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_Max_PtfmPitch_vc_mesh_over_Sideclmn_normalize_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #Plot the surface.
    surf = ax.plot_surface(Y_pntoon_upper_vec_mesh, Y_pntoon_lower_vec_mesh, Max_PtfmPitch_vc_mesh, cmap='Reds',
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_zlabel(r'$\theta_p^{\mathrm{max}} \,\, \mathrm{[deg]}$', fontsize=15)
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.1f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=18, azim=-166)
    ax.dist = 15  # Set the distance from the viewer to the axes
    #plt.show()
    fig.savefig(os.getcwd()+'/outputs/' +
                    'Max_PtfmPitch_vc_mesh_3D_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.35 )  
    
    #--------------------------------- STD Pltfrm Pitch-----------------------------------------
    
    # Normalize the data
    norm = Normalize(vmin=Std_PtfmPitch_vec_mesh.min(), vmax=Std_PtfmPitch_vec_mesh.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(Std_PtfmPitch_vec_mesh, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, Std_PtfmPitch_vec_mesh.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, Std_PtfmPitch_vec_mesh.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_title('$\mathrm{Std\,\,Ptfm\,\,Pitch\,\,[deg]}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(Std_PtfmPitch_vec_mesh.shape[0]):
        for j in range(Std_PtfmPitch_vec_mesh.shape[1]):
            ax.text(j, i, f'{Std_PtfmPitch_vec_mesh[i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'Std_PtfmPitch_vec_mesh_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55) 
    fig.savefig(os.getcwd()+'/outputs/' +
                    'Std_PtfmPitch_vec_mesh_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.55 )  
    
    # Assuming x and F are your arrays
    delta_Y_pntoon_upper_vec_mesh = np.gradient(Y_pntoon_upper_vec_mesh, axis=1)  # Compute the gradient of x along the second axis
    delta_Std_PtfmPitch_vec_mesh_over_up_Pntoon= np.gradient(Std_PtfmPitch_vec_mesh, axis=1) / delta_Y_pntoon_upper_vec_mesh  # Compute the derivative of F with respect to x
    delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize=delta_Std_PtfmPitch_vec_mesh_over_mainclmn*10.0/6.8 #D_baseline=10, thet_baseline=6.8

    # Normalize the data
    norm = Normalize(vmin=delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize.min(), vmax=-delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize.min())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=18)
    #ax.set_title('$\frac{\Delta \theta_p/\theta_p^0}{\Delta D_{\mathrm{Main}}/D_{\mathrm{Main}}^0}$', fontsize=15)
    ax.set_title(r'$\frac{{\Delta \sigma \theta_p/\theta_p^0}}{{\Delta D_{\mathrm{Main}}/D_{\mathrm{Main}}^0}}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=35, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize.shape[0]):
        for j in range(delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize.shape[1]):
            ax.text(j, i, f'{delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize[i, j]:.2f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_Std_PtfmPitch_vec_mesh_over_mainclmn_normalize_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  

    # Assuming x and F are your arrays
    delta_Y_pntoon_lower_vec_mesh = np.gradient(Y_pntoon_lower_vec_mesh, axis=0)  # Compute the gradient of x along the second axis
    delta_Std_PtfmPitch_vec_mesh_side_clmns = np.gradient(Std_PtfmPitch_vec_mesh, axis=0) / delta_Y_pntoon_lower_vec_mesh  # Compute the derivative of F with respect to x
    delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize=delta_Std_PtfmPitch_vec_mesh_side_clmns*12.5/6.8 #D_baseline=10, thet_baseline=6.8

    # Normalize the data
    norm = Normalize(vmin=delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize.min(), vmax=-30*delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize.min())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=18)
    #ax.set_title('$\frac{\Delta \theta_p/\theta_p^0}{\Delta D_{\mathrm{Main}}/D_{\mathrm{Main}}^0}$', fontsize=15)
    ax.set_title(r'$\frac{{\Delta \sigma \theta_p/\theta_p^0}}{{\Delta D_{\mathrm{Side}}/D_{\mathrm{Side}}^0}}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=35, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize.shape[0]):
        for j in range(delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize.shape[1]):
            ax.text(j, i, f'{delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize[i, j]:.2f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_Std_PtfmPitch_vec_mesh_side_clmns_normalize_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight' ,pad_inches=0.55)     

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #Plot the surface.
    surf = ax.plot_surface(Y_pntoon_upper_vec_mesh, Y_pntoon_lower_vec_mesh, Std_PtfmPitch_vec_mesh, cmap='Reds',
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_zlabel(r' Std - $\theta_p^{\mathrm{}max} \,\, \mathrm{[deg]}$', fontsize=15)
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.1f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=18, azim=-166)
    ax.dist = 15  # Set the distance from the viewer to the axes
    ##plt.show()
    fig.savefig(os.getcwd()+'/outputs/' +
                    'Std_PtfmPitch_vec_mesh_3D_clmns_normalize_upPntoonDiam_lowpNTOONdIAM.svg', format='svg', dpi=300, bbox_inches='tight', pad_inches=0.35 )  
    
    #--------------------------------- floatingse_structurall_mass -----------------------------------------
    
    # Normalize the data
    norm = Normalize(vmin=floatingse_structurall_mass_mesh.min(), vmax=floatingse_structurall_mass_mesh.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(floatingse_structurall_mass_mesh, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, floatingse_structurall_mass_mesh.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, floatingse_structurall_mass_mesh.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_title('$\mathrm{floating\,structural\,mass} \,\, [\mathrm{Kilo \,\,Tons}]$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(floatingse_structurall_mass_mesh.shape[0]):
        for j in range(floatingse_structurall_mass_mesh.shape[1]):
            ax.text(j, i, f'{floatingse_structurall_mass_mesh[i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'floatingse_structurall_mass_mesh_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'floatingse_structurall_mass_mesh_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight',pad_inches=0.55)  

        # Assuming x and F are your arrays
    delta_Y_pntoon_upper_vec_mesh = np.gradient(Y_pntoon_upper_vec_mesh, axis=1)  # Compute the gradient of x along the second axis
    delta_floatingse_structurall_mass_mesh_over_up_Pntoon= np.gradient(floatingse_structurall_mass_mesh, axis=1) / delta_Y_pntoon_upper_vec_mesh  # Compute the derivative of F with respect to x
    delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize=delta_floatingse_structurall_mass_mesh_over_mainclmn*10.0/6.8 #D_baseline=10, thet_baseline=6.8

    # Normalize the data
    norm = Normalize(vmin=delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize.min(), vmax=-delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize.min())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_title(r'$\frac{{\Delta m_{\mathrm{strct}}/m^0}}{{\Delta D_{\mathrm{Main}}/D_{\mathrm{Main}}^0}}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=35, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize.shape[0]):
        for j in range(delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize.shape[1]):
            ax.text(j, i, f'{delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize[i, j]:.2f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_floatingse_structurall_mass_mesh_over_mainclmn_normalize_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  

    # Assuming x and F are your arrays
    delta_Y_pntoon_lower_vec_mesh = np.gradient(Y_pntoon_lower_vec_mesh, axis=0)  # Compute the gradient of x along the second axis
    delta_floatingse_structurall_mass_mesh_side_clmns = np.gradient(floatingse_structurall_mass_mesh, axis=0) / delta_Y_pntoon_lower_vec_mesh  # Compute the derivative of F with respect to x
    delta_floatingse_structurall_mass_mesh_side_clmns_normalize=delta_floatingse_structurall_mass_mesh_side_clmns*12.5/6.8 #D_baseline=10, thet_baseline=6.8

    # Normalize the data
    norm = Normalize(vmin=delta_floatingse_structurall_mass_mesh_side_clmns_normalize.min(), vmax=delta_floatingse_structurall_mass_mesh_side_clmns_normalize.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(delta_floatingse_structurall_mass_mesh_side_clmns_normalize, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, delta_floatingse_structurall_mass_mesh_side_clmns_normalize.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, delta_floatingse_structurall_mass_mesh_side_clmns_normalize.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=18)
    #ax.set_title('$\frac{\Delta \theta_p/\theta_p^0}{\Delta D_{\mathrm{Main}}/D_{\mathrm{Main}}^0}$', fontsize=15)
    ax.set_title(r'$\frac{{\Delta m_{\mathrm{strct}}/m^0}}{{\Delta D_{\mathrm{Side}}/D_{\mathrm{Side}}^0}}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=35, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(delta_floatingse_structurall_mass_mesh_side_clmns_normalize.shape[0]):
        for j in range(delta_floatingse_structurall_mass_mesh_side_clmns_normalize.shape[1]):
            ax.text(j, i, f'{delta_floatingse_structurall_mass_mesh_side_clmns_normalize[i, j]:.2f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_floatingse_structurall_mass_mesh_side_clmns_normalize_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_floatingse_structurall_mass_mesh_side_clmns_normalize_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight' ,pad_inches=0.55)     

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #Plot the surface.
    surf = ax.plot_surface(Y_pntoon_upper_vec_mesh, Y_pntoon_lower_vec_mesh, floatingse_structurall_mass_mesh/1e6, cmap='Reds',
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_zlabel('$\mathrm{floating\,structural\,mass} \,\, [\mathrm{Kilo \,\,Tons}]$', fontsize=15)
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.1f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=18, azim=-166)
    ax.dist = 15  # Set the distance from the viewer to the axes
    #plt.show()
    fig.savefig(os.getcwd()+'/outputs/' +
                    'floatingse_structurall_mass_mesh_3D_upPntoonDiam_lowpNTOONdIAM.svg', format='svg', dpi=300, bbox_inches='tight', pad_inches=0.35 )  

    #--------------------------------- floatingse_platform_mass -----------------------------------------

    # Normalize the data
    norm = Normalize(vmin=floatingse_platform_mass_vec_mesh.min(), vmax=floatingse_platform_mass_vec_mesh.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(floatingse_platform_mass_vec_mesh, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, floatingse_platform_mass_vec_mesh.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, floatingse_platform_mass_vec_mesh.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_title('$\mathrm{platfrom\,mass} \,\, [\mathrm{Kilo \,\,Tons}]$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=30, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(floatingse_platform_mass_vec_mesh.shape[0]):
        for j in range(floatingse_platform_mass_vec_mesh.shape[1]):
            ax.text(j, i, f'{floatingse_platform_mass_vec_mesh[i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'floatingse_platform_mass_vec_mesh_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'floatingse_platform_mass_vec_mesh_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  

        # Assuming x and F are your arrays
    delta_Y_pntoon_upper_vec_mesh = np.gradient(Y_pntoon_upper_vec_mesh, axis=1)  # Compute the gradient of x along the second axis
    delta_floatingse_platform_mass_vec_mesh_over_up_Pntoon= np.gradient(floatingse_platform_mass_vec_mesh, axis=1) / delta_Y_pntoon_upper_vec_mesh  # Compute the derivative of F with respect to x
    delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize=delta_floatingse_platform_mass_vec_mesh_over_mainclmn*10.0/6.8 #D_baseline=10, thet_baseline=6.8

    # Normalize the data
    norm = Normalize(vmin=delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize.min(), vmax=-delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize.min())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_title(r'$\frac{{\Delta m_{\mathrm{platfrom}}/m^0}}{{\Delta D_{\mathrm{Main}}/D_{\mathrm{Main}}^0}}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=35, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize.shape[0]):
        for j in range(delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize.shape[1]):
            ax.text(j, i, f'{delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize[i, j]:.2f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_floatingse_platform_mass_vec_mesh_over_mainclmn_normalize_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  

    # Assuming x and F are your arrays
    delta_Y_pntoon_lower_vec_mesh = np.gradient(Y_pntoon_lower_vec_mesh, axis=0)  # Compute the gradient of x along the second axis
    delta_floatingse_platform_mass_vec_mesh_side_clmns = np.gradient(floatingse_platform_mass_vec_mesh, axis=0) / delta_Y_pntoon_lower_vec_mesh  # Compute the derivative of F with respect to x
    delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize=delta_floatingse_platform_mass_vec_mesh_side_clmns*12.5/6.8 #D_baseline=10, thet_baseline=6.8

    # Normalize the data
    norm = Normalize(vmin=delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize.min(), vmax=delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
    x_indices = np.arange(0, delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{Y_pntoon_upper_vec_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
    y_indices = np.arange(0, delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=18)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=18)
    #ax.set_title('$\frac{\Delta \theta_p/\theta_p^0}{\Delta D_{\mathrm{Main}}/D_{\mathrm{Main}}^0}$', fontsize=15)
    ax.set_title(r'$\frac{{\Delta m_{\mathrm{platfrom}}/m^0}}{{\Delta D_{\mathrm{Side}}/D_{\mathrm{Side}}^0}}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(Y_pntoon_upper_vec_mesh[0,:]==10)[0][0],np.where(Y_pntoon_lower_vec_mesh[:,0]==12.5)[0][0] , 's', markersize=35, markerfacecolor='none', markeredgecolor='black', markeredgewidth =2)
    for i in range(delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize.shape[0]):
        for j in range(delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize.shape[1]):
            ax.text(j, i, f'{delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize[i, j]:.2f}', ha='center', va='center', color='black', fontsize=12)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight' ,pad_inches=0.55)  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'delta_floatingse_platform_mass_vec_mesh_side_clmns_normalize_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight' ,pad_inches=0.55 )     
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #Plot the surface.
    surf = ax.plot_surface(Y_pntoon_upper_vec_mesh, Y_pntoon_lower_vec_mesh, floatingse_platform_mass_vec_mesh/1e6, cmap='Reds',
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_ylabel('$D_\mathrm{pntn- low}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_xlabel('$D_\mathrm{pntn- up}\,\, \mathrm{[m]}$',, fontsize=15)
    ax.set_zlabel('$\mathrm{floating\,Platform\,mass} \,\, [\mathrm{Kilo \,\,Tons}]$', fontsize=15)
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.1f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=18, azim=-166)
    ax.dist = 15  # Set the distance from the viewer to the axes
    #plt.show()
    fig.savefig(os.getcwd()+'/outputs/' +
                    'floatingse_platform_mass_vec_mesh_3D_upPntoonDiam_lowpNTOONdIAM.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.35 )  
    fig.savefig(os.getcwd()+'/outputs/' +
                    'floatingse_platform_mass_vec_mesh_3D_upPntoonDiam_lowpNTOONdIAM.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.35 )  
