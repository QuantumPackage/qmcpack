program qmcpack
  implicit none
  BEGIN_DOC
! Generates a file for QMCPACK 
  END_DOC

  integer :: i,j
  read_wf = .True.
  TOUCH read_wf
  call save_wavefunction_qmcpack
  if (is_complex) then

    if (qmc_do_reorder_occ) then
      call qmc_set_order_occ_kpts
    else
      print*,'WARNING: MO coefs will not be reordered by occupation number'
      print*,'         using ordering from qmcpack/mo_coef_reorder_idx_kpts.gz'

      !call qmc_set_order_energy_kpts
    endif
    if (.True.) then
      complex*16, allocatable :: tmp_coef_ordered(:,:)
      integer :: ii,jj

      allocate(tmp_coef_ordered(ao_num,mo_num))
      tmp_coef_ordered = (0.d0,0.d0)
      do ii=1,mo_num
        jj=mo_coef_reorder_idx_kpts(ii)
        do i=1,ao_num
          tmp_coef_ordered(i,ii) = mo_coef_complex(i,jj)
        enddo
      enddo
      call ezfio_set_qmcpack_mo_coef_complex_reordered(tmp_coef_ordered)
    endif
    !TODO: do we need to normalize mo_coef_complex?
  else
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
  endif
  !TODO: fix python for complex
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
    if (is_complex) then
      call save_wavefunction_general_complex(N_det,N_states,&
                  psi_det_sorted,size(psi_coef_sorted_complex,1),psi_coef_sorted_complex)
    else
      call save_wavefunction_general(N_det,N_states,psi_det_sorted,size(psi_coef_sorted,1),psi_coef_sorted)
    endif
  endif
end


