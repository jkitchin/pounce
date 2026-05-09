! fortran_open_fixed.f90
! Wrappers for Fortran file I/O called from Rust via C FFI.
! CUTEst reads OUTSDIF.d via Fortran unit numbers, so we need
! Fortran OPEN/CLOSE to set up the unit before calling CUTEst routines.
!
! Uses bind(c) so the caller does not need to pass hidden string lengths.

subroutine fortran_open_fixed(funit, fname, ierr) bind(c, name='fortran_open_fixed_')
  use iso_c_binding, only: c_int, c_char, c_null_char
  implicit none
  integer(c_int), intent(in)  :: funit
  character(kind=c_char), intent(in) :: fname(*)
  integer(c_int), intent(out) :: ierr
  character(len=4096) :: fpath
  integer :: i

  ! Copy C string (null-terminated) into Fortran string
  fpath = ' '
  do i = 1, 4096
    if (fname(i) == c_null_char) exit
    fpath(i:i) = fname(i)
  end do

  ierr = 0
  open(unit=funit, file=trim(fpath), status='old', form='formatted', iostat=ierr)
end subroutine fortran_open_fixed

subroutine fortran_close(funit, ierr) bind(c, name='fortran_close_')
  use iso_c_binding, only: c_int
  implicit none
  integer(c_int), intent(in)  :: funit
  integer(c_int), intent(out) :: ierr

  ierr = 0
  close(unit=funit, iostat=ierr)
end subroutine fortran_close
