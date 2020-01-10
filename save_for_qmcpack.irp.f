program qmcpack
  implicit none
  BEGIN_DOC
! Generates a file for QMCPACK 
  END_DOC

  integer :: i,j
  read_wf = .True.
  TOUCH read_wf
  call save_wavefunction_qmcpack
  do j=1,ao_prim_num_max
    do i=1,ao_num
      ao_coef(i,j) = ao_coef(i,j) * ao_coef_normalization_factor(i)
    enddo
  enddo
  call ezfio_set_ao_basis_ao_coef(ao_coef)
  do j=1,mo_num
    do i=1,ao_num
      mo_coef(i,j) *= 1.d0/ao_coef_normalization_factor(i)
    enddo
  enddo
  call save_mos
  call system('rm '//trim(ezfio_filename)//'/mo_basis/ao_md5')
  call system('$QP_ROOT/src/qmcpack/qp_convert_qmcpack_to_ezfio.py '//trim(ezfio_filename))

end

subroutine save_wavefunction_qmcpack
  implicit none
  use bitmasks
  BEGIN_DOC
  !  Save the wave function into the |EZFIO| file
  END_DOC

  if (N_det < N_states) then
    return
  endif
  if (mpi_master) then
    call save_wavefunction_general(N_det,N_states,psi_det_sorted,size(psi_coef_sorted,1),psi_coef_sorted)
  endif
end

